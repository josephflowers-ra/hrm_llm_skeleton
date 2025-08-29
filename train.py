import argparse
import math
import os
import random
import time

import torch
from torch.utils.data import DataLoader

from model import HRMController, HRMConfig
from reasoning_gym_wrapper import build_reasoning_dataset

def set_seed(s):
    random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)

def build_dataloaders(task_name, batch_size, max_train=2000, max_eval=200):
    ds_train = build_reasoning_dataset(name=task_name, split="train", n=max_train)
    ds_eval = build_reasoning_dataset(name=task_name, split="eval", n=max_eval)
    return ds_train, ds_eval

def evaluate(model, dl, device, max_batches=10):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for b_idx, batch in enumerate(dl):
            collated = model.collate(batch)
            for k in ("input_ids","attention_mask","labels"):
                collated[k] = collated[k].to(device)

            # Forward once to get logits with last segment injection
            last_hidden = model.forward_llm_hidden(collated["input_ids"], collated["attention_mask"])
            x_pool = model.pool_tokens(last_hidden, collated["attention_mask"], collated["prompt_lengths"])
            x_tilde = model.x_proj(x_pool)

            zH_list = model.hrm_segments(x_tilde, segments=model.hrm_cfg.segments, inner_T=model.hrm_cfg.inner_T)
            zH = zH_list[-1]

            injected = model.injector(last_hidden, zH)
            logits = model.logits_from_injected(injected)

            # Greedy decode proxy: take argmax over target region
            pred_ids = logits.argmax(dim=-1)
            for i, item in enumerate(batch):
                Lp = collated["prompt_lengths"][i]
                pred_text = model.tokenizer.decode(pred_ids[i, Lp:], skip_special_tokens=True)
                is_ok = item["verify"](pred_text) if callable(item["verify"]) else False
                correct += int(is_ok); total += 1

            if b_idx+1 >= max_batches: break

    acc = correct / max(1,total)
    return {"eval_acc_proxy": acc, "n": total}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_name", type=str, default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    ap.add_argument("--task", type=str, default="toy_add")
    ap.add_argument("--segments", type=int, default=3)
    ap.add_argument("--inner_T", type=int, default=3)
    ap.add_argument("--use_act", action="store_true", default=False)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--max_steps", type=int, default=200)
    ap.add_argument("--eval_every", type=int, default=50)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    print(f"[INFO] Using device: {args.device}")
    set_seed(args.seed)

    # Build model
    hrm_cfg = HRMConfig(inner_T=args.inner_T, segments=args.segments, use_act=args.use_act)
    model = HRMController(model_name=args.model_name, hrm_cfg=hrm_cfg).to(args.device)

    # Data
    ds_train, ds_eval = build_dataloaders(args.task, args.batch_size)
    dl_train = DataLoader(ds_train, batch_size=args.batch_size, shuffle=True, drop_last=True, collate_fn=lambda b: b)
    dl_eval = DataLoader(ds_eval, batch_size=args.batch_size, shuffle=False, drop_last=False, collate_fn=lambda b: b)

    # Optimizer
    optim = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)
    print(f"[INFO] Trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    step = 0
    model.train()
    t0 = time.time()

    while step < args.max_steps:
        for batch in dl_train:
            collated = model.collate(batch)
            for k in ("input_ids","attention_mask","labels"):
                collated[k] = collated[k].to(args.device)

            loss, metrics = model.training_step(collated, segments=args.segments, inner_T=args.inner_T, use_act=args.use_act)
            optim.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()

            step += 1
            if step % 10 == 0:
                seg_losses = ", ".join(f"{x:.3f}" for x in metrics['segment_losses'])
                print(f"[STEP {step}] loss={metrics['loss']:.4f} | seg_losses=[{seg_losses}]")

            if step % args.eval_every == 0:
                eval_stats = evaluate(model, dl_eval, args.device, max_batches=5)
                dt = time.time() - t0
                print(f"[EVAL {step}] acc_proxy={eval_stats['eval_acc_proxy']:.3f} on n={eval_stats['n']} | elapsed={dt/60:.1f}m")

            if step >= args.max_steps:
                break

    print("[DONE] Training finished.")

if __name__ == "__main__":
    main()
