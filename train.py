#!/usr/bin/env python3
"""
Minimal trainer for the HRM + LLM hybrid, with spot-check logging.

Adds:
- --injector {grb,cab} to choose GRB vs CAB
- --save_dir / --save_every / --resume (saves optimizer state too)
- --eval_batches to control quick proxy eval loop
- --log_samples_every to print example prompt/target/pred triples
- --max_new_tokens to cap decoded length in spot-checks
"""

import argparse, os, random, time, torch
from torch.utils.data import DataLoader

from model import HRMController, HRMConfig
from reasoning_gym_wrapper import build_reasoning_dataset


# ---------------------------
# Reproducibility
# ---------------------------
def set_seed(s: int):
    random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)
    # For stricter determinism, you can uncomment:
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False


# ---------------------------
# Lightweight proxy evaluation:
# Uses dataset's verify() to score greedy outputs.
# ---------------------------
def evaluate(model: HRMController, dl, device: str, max_batches: int = 10):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for b_idx, batch in enumerate(dl):
            # Tokenize/label
            col = model.collate(batch)
            for k in ("input_ids", "attention_mask", "labels"):
                col[k] = col[k].to(device)

            # 1) Frozen LLM hidden states
            last_hidden = model.forward_llm_hidden(col["input_ids"], col["attention_mask"])

            # 2) Pool prompt → project → HRM loop (take last zH)
            x_pool = model.pool_tokens(last_hidden, col["attention_mask"], col["prompt_lengths"])
            x_tilde = model.x_proj(x_pool)
            zH = model.hrm_segments(
                x_tilde, segments=model.hrm_cfg.segments, inner_T=model.hrm_cfg.inner_T
            )[-1]

            # 3) Inject → LM head → logits → greedy decode over target region
            logits = model.logits_from_injected(model.injector(last_hidden, zH))
            pred_ids = logits.argmax(-1)

            for i, item in enumerate(batch):
                Lp = col["prompt_lengths"][i]
                text = model.tokenizer.decode(pred_ids[i, Lp:], skip_special_tokens=True)
                ok = item.get("verify", None)
                correct += int(callable(ok) and ok(text))
                total += 1

            if b_idx + 1 >= max_batches:
                break

    return {"eval_acc_proxy": correct / max(1, total), "n": total}


# ---------------------------
# Utility: pretty print a few samples (prompt/target/pred) + zH norm
# ---------------------------
def spot_check_samples(model: HRMController, dl_eval, device: str, max_new_tokens: int, n_print: int = 3):
    model.eval()
    with torch.no_grad():
        try:
            batch = next(iter(dl_eval))
        except StopIteration:
            return
        col = model.collate(batch)
        for k in ("input_ids", "attention_mask", "labels"):
            col[k] = col[k].to(device)

        last_hidden = model.forward_llm_hidden(col["input_ids"], col["attention_mask"])
        x_pool = model.pool_tokens(last_hidden, col["attention_mask"], col["prompt_lengths"])
        x_tilde = model.x_proj(x_pool)
        zH = model.hrm_segments(
            x_tilde, segments=model.hrm_cfg.segments, inner_T=model.hrm_cfg.inner_T
        )[-1]
        inj = model.injector(last_hidden, zH)
        logits = model.logits_from_injected(inj)
        pred_ids = logits.argmax(-1)

        # Debug latent magnitude
        z_norm = torch.norm(zH, dim=-1).mean().item()
        print(f"\n[SPOT CHECK] mean ||z_H|| = {z_norm:.3f}")

        # Print a few examples
        print("[SPOT CHECK] Showing up to", n_print, "examples:")
        for i, item in enumerate(batch[:n_print]):
            Lp = col["prompt_lengths"][i]
            prompt_ids = col["input_ids"][i, :Lp]
            target_ids = col["labels"][i, Lp:]
            # Replace ignore_index (-100) with pad for readable decode
            target_ids = target_ids.masked_fill(target_ids == model.hrm_cfg.vocab_ignore_index,
                                               model.tokenizer.pad_token_id or 0)

            # Cap decode lengths for readability
            prompt_txt = model.tokenizer.decode(prompt_ids[:max_new_tokens],
                                                skip_special_tokens=True)
            target_txt = model.tokenizer.decode(target_ids[:max_new_tokens],
                                                skip_special_tokens=True)
            pred_txt   = model.tokenizer.decode(pred_ids[i, Lp: Lp + max_new_tokens],
                                                skip_special_tokens=True)

            print(f"\n-- Example {i+1} --")
            print("Prompt:", prompt_txt.strip())
            print("Target:", target_txt.strip())
            print("Pred:  ", pred_txt.strip())
    model.train()


# ---------------------------
# Main
# ---------------------------
def main():
    ap = argparse.ArgumentParser()
    # Base model + task
    ap.add_argument("--model_name", type=str, default="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
                    help="HF name or local path to a CausalLM (frozen).")
    ap.add_argument("--task", type=str, default="toy_add", help="reasoning_gym task or 'toy_add'.")

    # HRM depth controls
    ap.add_argument("--segments", type=int, default=3, help="Number of HRM segments (H updates).")
    ap.add_argument("--inner_T", type=int, default=3, help="Number of L micro-steps per segment.")

    # Injector choice
    ap.add_argument("--injector", type=str, choices=("grb", "cab"), default="grb",
                    help="GRB = gated residual bias; CAB = cross-attention bridge (gated).")

    # Optional ACT + temperature
    ap.add_argument("--use_act", action="store_true", default=False, help="Enable ACT (halt head) training.")
    ap.add_argument("--act_penalty", type=float, default=0.0, help="Compute penalty to bias fewer segments.")
    ap.add_argument("--temperature", type=float, default=1.0, help="Logit temperature (proxy for stablemax CE).")

    # Optimization + runtime
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--max_steps", type=int, default=200)
    ap.add_argument("--eval_every", type=int, default=50)
    ap.add_argument("--eval_batches", type=int, default=5)

    # I/O
    ap.add_argument("--save_dir", type=str, default="checkpoints")
    ap.add_argument("--save_every", type=int, default=1000, help="Save every N steps (e.g., 500 or 1000).")
    ap.add_argument("--resume", type=str, default="", help="Path to .pt checkpoint to resume.")

    # Spot-check logs
    ap.add_argument("--log_samples_every", type=int, default=500,
                    help="Steps between printing example prompt/target/pred triples.")
    ap.add_argument("--max_new_tokens", type=int, default=64,
                    help="Max tokens to decode for targets/preds in spot-checks.")

    # System
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    args = ap.parse_args()

    # Build HRM config (flip use_cab per injector flag)
    use_cab = (args.injector.lower() == "cab")
    hrm_cfg = HRMConfig(
        inner_T=args.inner_T,
        segments=args.segments,
        use_act=args.use_act,
        use_cab=use_cab
    )

    # Boot
    print(f"[INFO] Device: {args.device}")
    print(f"[INFO] Injector: {args.injector.upper()} | segments={args.segments} | inner_T={args.inner_T}")
    set_seed(args.seed)

    # Model (LLM is frozen; HRM + injector trainable)
    model = HRMController(args.model_name, hrm_cfg).to(args.device)

    # Resume (optional)
    os.makedirs(args.save_dir, exist_ok=True)
    optim = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr, betas=(0.9, 0.95), weight_decay=0.1
    )
    if args.resume:
        ckpt = torch.load(args.resume, map_location=args.device)
        model.load_trainable_state_dict(ckpt["state"])
        if "optim" in ckpt:
            optim.load_state_dict(ckpt["optim"])
        print(f"[INFO] Resumed from {args.resume} (step={ckpt.get('step')})")

    print(f"[INFO] Trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    # Data
    ds_train = build_reasoning_dataset(args.task, "train", n=2000)
    ds_eval  = build_reasoning_dataset(args.task, "eval",  n=200)
    dl_train = DataLoader(ds_train, batch_size=args.batch_size, shuffle=True,
                          drop_last=True,  collate_fn=lambda b: b)
    dl_eval  = DataLoader(ds_eval,  batch_size=args.batch_size, shuffle=False,
                          drop_last=False, collate_fn=lambda b: b)

    # Train
    step = 0
    model.train()
    t0 = time.time()

    while step < args.max_steps:
        for batch in dl_train:
            col = model.collate(batch)
            for k in ("input_ids", "attention_mask", "labels"):
                col[k] = col[k].to(args.device)

            loss, metrics = model.training_step(
                col,
                segments=args.segments,
                inner_T=args.inner_T,
                use_act=args.use_act,
                temperature=args.temperature,
                act_penalty=args.act_penalty
            )

            optim.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()
            step += 1

            if step % 10 == 0:
                seg_losses = [round(x, 3) for x in metrics['segment_losses']]
                q_losses   = [round(x, 3) for x in metrics['q_losses']]
                print(f"[STEP {step}] loss={metrics['loss']:.4f} | seg_losses={seg_losses} | q_losses={q_losses}")

            if step % args.eval_every == 0:
                ev = evaluate(model, dl_eval, args.device, max_batches=args.eval_batches)
                dt = time.time() - t0
                print(f"[EVAL {step}] acc_proxy={ev['eval_acc_proxy']:.3f} on n={ev['n']} | elapsed={dt/60:.1f}m")

            # Spot-check a few samples (prompt/target/pred) and z_H norm
            if step % args.log_samples_every == 0:
                spot_check_samples(model, dl_eval, args.device, args.max_new_tokens, n_print=3)

            # Save lightweight checkpoints (model + optimizer)
            if (step % args.save_every == 0) or (step >= args.max_steps):
                path = os.path.join(args.save_dir, f"hrm_step{step}.pt")
                torch.save({"step": step,
                            "state": model.trainable_state_dict(),
                            "optim": optim.state_dict()}, path)
                print(f"[CKPT] Saved {path}")

            if step >= args.max_steps:
                break

    print("[DONE]")


if __name__ == "__main__":
    main()
