#!/usr/bin/env python3
"""
Minimal trainer for the HRM + LLM hybrid.

Key additions vs original:
- --injector {grb,cab} to choose the conditioning path from HRM → LLM
- More comments explaining each step of the pipeline
- Optional --eval_batches to control the quick proxy eval loop
"""

import argparse, random, time, torch
from torch.utils.data import DataLoader
from model import HRMController, HRMConfig
from reasoning_gym_wrapper import build_reasoning_dataset


# ---------------------------
# Reproducibility helpers
# ---------------------------
def set_seed(s: int):
    random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)
    # If you want stricter determinism, uncomment below (slower on some GPUs):
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False


# ---------------------------
# Lightweight proxy evaluation
# - Uses dataset's programmatic verifier (if provided) to score greedy outputs
# - Runs with model frozen/no_grad
# ---------------------------
def evaluate(model: HRMController, dl, device: str, max_batches: int = 10):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for b_idx, batch in enumerate(dl):
            # Tokenize and build labels (prompts masked out with -100)
            col = model.collate(batch)
            for k in ("input_ids", "attention_mask", "labels"):
                col[k] = col[k].to(device)

            # 1) Frozen LLM forward to get last token hidden states
            last_hidden = model.forward_llm_hidden(col["input_ids"], col["attention_mask"])

            # 2) Pool only the prompt region (mean over prompt tokens) → project to HRM dim
            x_pool = model.pool_tokens(last_hidden, col["attention_mask"], col["prompt_lengths"])
            x_tilde = model.x_proj(x_pool)

            # 3) Run HRM recurrence; take zH from the LAST segment for eval
            zH = model.hrm_segments(
                x_tilde,
                segments=model.hrm_cfg.segments,
                inner_T=model.hrm_cfg.inner_T
            )[-1]

            # 4) Inject HRM state into LLM hidden states, get logits from LM head
            logits = model.logits_from_injected(model.injector(last_hidden, zH))

            # 5) Greedy decode only over the target region (after prompt length)
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
# Main training entry
# ---------------------------
def main():
    ap = argparse.ArgumentParser()
    # Base model + task
    ap.add_argument("--model_name", type=str, default="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
                    help="HF name or local path to a CausalLM (frozen as 'speaker').")
    ap.add_argument("--task", type=str, default="toy_add",
                    help="reasoning_gym task name or 'toy_add' fallback.")

    # HRM depth controls
    ap.add_argument("--segments", type=int, default=3, help="Number of HRM segments (H updates).")
    ap.add_argument("--inner_T", type=int, default=3, help="Number of L micro-steps per segment.")

    # Injector choice (HRM → LLM)
    ap.add_argument("--injector", type=str, choices=("grb", "cab"), default="grb",
                    help="GRB = gated residual bias; CAB = cross-attention bridge.")

    # Optional ACT halting + temperature
    ap.add_argument("--use_act", action="store_true", default=False, help="Enable ACT (halt head) training.")
    ap.add_argument("--act_penalty", type=float, default=0.0, help="Small compute penalty to bias fewer segments.")
    ap.add_argument("--temperature", type=float, default=1.0, help="Logit temperature (proxy for stablemax CE).")

    # Optimization + runtime
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--max_steps", type=int, default=200)
    ap.add_argument("--eval_every", type=int, default=50)
    ap.add_argument("--eval_batches", type=int, default=5, help="How many batches to use in each quick eval.")
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    # Build HRM config based on CLI (injector flag flips use_cab)
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

    # Model (LLM is frozen; HRM + injector are trainable)
    model = HRMController(args.model_name, hrm_cfg).to(args.device)

    # Data
    ds_train = build_reasoning_dataset(args.task, "train", n=2000)
    ds_eval  = build_reasoning_dataset(args.task, "eval",  n=200)
    dl_train = DataLoader(ds_train, batch_size=args.batch_size, shuffle=True,
                          drop_last=True,  collate_fn=lambda b: b)
    dl_eval  = DataLoader(ds_eval,  batch_size=args.batch_size, shuffle=False,
                          drop_last=False, collate_fn=lambda b: b)

    # Optimizer (trainable params = HRM + injector only)
    optim = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr, betas=(0.9, 0.95), weight_decay=0.1
    )
    print(f"[INFO] Trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    # ---------------------------
    # Training loop
    # ---------------------------
    step = 0
    model.train()
    t0 = time.time()

    while step < args.max_steps:
        for batch in dl_train:
            # Build tokenized batch + send to device
            col = model.collate(batch)
            for k in ("input_ids", "attention_mask", "labels"):
                col[k] = col[k].to(args.device)

            # One training step with deep supervision over segments
            loss, metrics = model.training_step(
                col,
                segments=args.segments,
                inner_T=args.inner_T,
                use_act=args.use_act,
                temperature=args.temperature,
                act_penalty=args.act_penalty
            )

            # Standard optimizer step
            optim.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()
            step += 1

            # Periodic logging
            if step % 10 == 0:
                seg_losses = [round(x, 3) for x in metrics['segment_losses']]
                q_losses   = [round(x, 3) for x in metrics['q_losses']]
                print(f"[STEP {step}] loss={metrics['loss']:.4f} | seg_losses={seg_losses} | q_losses={q_losses}")

            # Quick proxy eval (verifier-based accuracy)
            if step % args.eval_every == 0:
                ev = evaluate(model, dl_eval, args.device, max_batches=args.eval_batches)
                dt = time.time() - t0
                print(f"[EVAL {step}] acc_proxy={ev['eval_acc_proxy']:.3f} on n={ev['n']} | elapsed={dt/60:.1f}m")

            if step >= args.max_steps:
                break

    print("[DONE]")


if __name__ == "__main__":
    main()
