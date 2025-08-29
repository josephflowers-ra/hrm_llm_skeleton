
import argparse, random, time, torch
from torch.utils.data import DataLoader
from model import HRMController, HRMConfig
from reasoning_gym_wrapper import build_reasoning_dataset

def set_seed(s): random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)

def evaluate(model, dl, device, max_batches=10):
    model.eval(); correct=0; total=0
    with torch.no_grad():
        for b_idx, batch in enumerate(dl):
            col=model.collate(batch)
            for k in ("input_ids","attention_mask","labels"): col[k]=col[k].to(device)
            last_hidden=model.forward_llm_hidden(col["input_ids"], col["attention_mask"])
            x_pool=model.pool_tokens(last_hidden, col["attention_mask"], col["prompt_lengths"]); x_tilde=model.x_proj(x_pool)
            zH=model.hrm_segments(x_tilde, segments=model.hrm_cfg.segments, inner_T=model.hrm_cfg.inner_T)[-1]
            logits=model.logits_from_injected(model.injector(last_hidden, zH))
            pred_ids=logits.argmax(-1)
            for i,item in enumerate(batch):
                Lp=col["prompt_lengths"][i]; text=model.tokenizer.decode(pred_ids[i,Lp:], skip_special_tokens=True)
                ok=item.get("verify", None); correct += int(callable(ok) and ok(text)); total+=1
            if b_idx+1 >= max_batches: break
    return {"eval_acc_proxy": correct/max(1,total), "n": total}

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--model_name", type=str, default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    ap.add_argument("--task", type=str, default="toy_add")
    ap.add_argument("--segments", type=int, default=3)
    ap.add_argument("--inner_T", type=int, default=3)
    ap.add_argument("--use_act", action="store_true", default=False)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--act_penalty", type=float, default=0.0)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--max_steps", type=int, default=200)
    ap.add_argument("--eval_every", type=int, default=50)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args=ap.parse_args()

    print(f"[INFO] Device: {args.device}"); set_seed(args.seed)
    model=HRMController(args.model_name, HRMConfig(inner_T=args.inner_T, segments=args.segments, use_act=args.use_act)).to(args.device)
    ds_train=build_reasoning_dataset(args.task, "train", n=2000); ds_eval=build_reasoning_dataset(args.task, "eval", n=200)
    dl_train=DataLoader(ds_train, batch_size=args.batch_size, shuffle=True, drop_last=True, collate_fn=lambda b: b)
    dl_eval=DataLoader(ds_eval, batch_size=args.batch_size, shuffle=False, drop_last=False, collate_fn=lambda b: b)
    optim=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, betas=(0.9,0.95), weight_decay=0.1)
    print(f"[INFO] Trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    step=0; model.train(); t0=time.time()
    while step < args.max_steps:
        for batch in dl_train:
            col=model.collate(batch)
            for k in ("input_ids","attention_mask","labels"): col[k]=col[k].to(args.device)
            loss, metrics=model.training_step(col, segments=args.segments, inner_T=args.inner_T, use_act=args.use_act, temperature=args.temperature, act_penalty=args.act_penalty)
            optim.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); optim.step()
            step += 1
            if step % 10 == 0:
                print(f"[STEP {step}] loss={metrics['loss']:.4f} | seg_losses={[round(x,3) for x in metrics['segment_losses']]} | q_losses={[round(x,3) for x in metrics['q_losses']]}")
            if step % args.eval_every == 0:
                ev=evaluate(model, dl_eval, args.device, max_batches=5); dt=time.time()-t0
                print(f"[EVAL {step}] acc_proxy={ev['eval_acc_proxy']:.3f} on n={ev['n']} | elapsed={dt/60:.1f}m")
            if step >= args.max_steps: break
    print("[DONE]")

if __name__=="__main__": main()
