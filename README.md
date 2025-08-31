
---




---
            ┌──────────────────────────────────────────────────────────┐
            │                        Input Text                        │
            │                (prompt; decode settings)                 │
            │            T=1.0 | top_p=0.0 | top_k=0 (greedy)          │
            └───────────────┬──────────────────────────────────────────┘
                            │
                            ▼
            ┌──────────────────────────────────────────────────────────┐
            │                  Frozen LLM (encoder)                    │
            │  - Run prompt (and growing decode context)               │
            │  - Collect last hidden states h ∈ ℝ^{B×T×D_model}        │
            └───────────────┬──────────────────────────────────────────┘
                            │
                 Pool ONLY the prompt region per sample
                    (mean + last) → linear mix → x_pool
                            │
                            ▼
            ┌──────────────────────────────────────────────────────────┐
            │          Project to HRM width: x̃ = W_proj(x_pool)        │
            │                    x̃ ∈ ℝ^{B×1×d_h}                       │
            └───────────────┬──────────────────────────────────────────┘
                            │
                            ▼
         ┌──────────────────┴───────────────────────────────────────┐
         │            HRM Reasoning Loop                            │
         │     (low-level fast L; high-level H)                     │
         │                                                          │
         │  for segment m = 1..M (or until ACT halts):              │   
         │   ┌───────────────────────────────────────────────────┐  │
         │   │   L-block (fast)                                  │  │
         │   │   - do inner_T-1 micro-steps (no_grad)            │  │
         │   │   - 1 final micro-step (with grad)                │  │
         │   └───────────┬───────────────────────────────────────┘  │
         │               │                                          │
         │   ┌───────────▼───────────────────────────────────────┐  │
         │   │   H-block (slow)                                  │  │
         │   │   - update z_H from z_L                           │  │
         │   │   - RMSNorm on z_H                                │  │
         │   └───────────┬───────────────────────────────────────┘  │
         │               │                                          │
         │      (optional) ACT / q-head on z_H                      │
         │               │                                          │
         │      if ACT says “halt”: break                           │
         │      else: detach(z_H, z_L) and continue                 │
         └──────────────────────────────────────────────────────────┘
                            │
                            │   final z_H (shape: B×1×d_h)
                            ▼
            ┌──────────────────────────────────────────────────────────┐
            │                 Injector Module (LLM side)               │
            │  Option A: GRB (z_H → bias in hidden space, gated)       │
            │  Option B: CAB with multi-token memory (m≥1), gated      │
            │   - Build z_H-derived memory bank (B×m×D_model)          │
            │   - Cross-attend from token states to memory             │
            │   - Residual add with sigmoid(gate)                      │
            └───────────────┬──────────────────────────────────────────┘
                            │
                    Conditioned hidden states h'
                            │
                            ▼
            ┌──────────────────────────────────────────────────────────┐
            │        (if present) Final Norm of base LLM               │
            │  e.g., LLaMA/Mistral: model.norm; NeoX: final_layer_norm │
            └───────────────┬──────────────────────────────────────────┘
                            │
                            ▼
            ┌──────────────────────────────────────────────────────────┐
            │                LLM LM Head → logits (B×T×V)              │
            │  + Gated vocab-bias from z_H (tiny linear, optional)     │
            │      logits += σ(vocab_gate) · W_vocab(z_H)              │
            └───────────────┬──────────────────────────────────────────┘
                            │
                    Select next token (greedy / sampling)
                            │
                            ▼
            ┌──────────────────────────────────────────────────────────┐
            │            Append token, update context T                │
            │        If EOS reached or max_len: stop decode            │
            └──────────────────────────────────────────────────────────┘




---


Love it—here’s an updated `README.md` that matches your current code and training setup (multi-token CAB, vocab-bias head, EOS handling, eval wiring, etc.). It also adds quickstarts, new CLI flags, troubleshooting, and checkpoint notes.

---

# HRM-LLM Hybrid (Frozen LLM + HRM Controller)

## Overview

This repo augments small, open-weight language models (TinyLlama, Qwen2.5, etc.) with a **Hierarchical Reasoning Model (HRM)** controller. The LLM stays **frozen** (no finetuning). A lightweight HRM runs **latent reasoning** on pooled prompt features and then **steers** the LLM’s output via:

* a **Cross-Attention Bridge (CAB)** with **multi-token memory** derived from the HRM state, and
* a tiny, gated **vocab-bias head** that adds a learned bias over the vocabulary at decode time.

This hybrid lets the LLM remain the **fluent communicator**, while HRM performs **silent computation** in hidden space.

What you get:

* HRM H/L blocks (fast/slow recurrence, hierarchical convergence).
* Injectors: **GRB** (gated residual bias) and **CAB** (multi-token cross-attn).
* Deep supervision with **O(1)** memory (detach between segments; one-step grads).
* Optional **ACT** halting head.
* Reasoning-Gym adapter + minimal training harness.

---

## Background: HRM in a Nutshell

HRM runs two timescales of recurrence:

* **L-module (fast)** takes several micro-steps per segment to find a local fixed point.
* **H-module (slow)** updates once per segment to guide L.

Training uses a one-step approximation (no BPTT through time), applies loss each segment, and **detaches** between segments—keeping memory usage constant.

Why this matters:

* **Compact latent reasoning** (no long token chains).
* **Inference scaling**: increase segments at eval to deepen reasoning without retraining.
* **Data-efficient**: useful on structured math/logical tasks.

---

## What’s New vs. the Old Skeleton

**Controller & decoding**

* Mixed prompt pooling (**mean + last**) → projection to HRM width.
* **Final norm** is applied (when the base model expects it) before the LM head.
* **EOS is appended** to each target to teach “answer → stop”.

**Stronger steering**

* **CAB** now supports **multi-token memory** (`mem_tokens`), not just a single token.
* **Vocab-bias head**: a tiny linear from HRM state (`z_H`) into the vocab logits, behind a sigmoid gate.
* Both are **gated** (sigmoid) so you can tune their strength.

**Eval wiring**

* During eval/spot-checks, logits are produced with
  `model.logits_from_injected(..., zH=zH)` so the **vocab-bias head** is active (this was crucial).

---

## Repository Layout

* **`hrm_blocks.py`** – RMSNorm, SwiGLU MLP, SelfAttention, TransformerBlock, `HBlock` / `LBlock`.
* **`injector.py`** –

  * `InjectorGRB`: gated residual bias (z\_H → hidden bias).
  * `CrossAttentionBridge`: **multi-token** z\_H memory with cross-attention (default `mem_tokens=4`), gated residual.
* **`model.py`** – Frozen HF CausalLM + HRM controller: pooling, recurrence, injection, vocab-bias head, collator, trainable state helpers.
* **`reasoning_gym_wrapper.py`** – Adapter that builds datasets & verifiers for tasks like `basic_arithmetic`, `gsm_symbolic`, `chain_sum`, `simple_equations`, etc.
* **`train.py`** – Minimal trainer with eval/spot-checks, CSV logging, checkpoints, CLI flags.

---

## Installation

```bash
pip install -r requirements.txt
# typical:
# torch, transformers, accelerate, datasets, einops, reasoning-gym
```

You’ll also need a local or HF-hub base model path, e.g. `TinyLlama/TinyLlama-1.1B-Chat-v1.0` or your local `./tiny-rl-sft`.

---

## Quickstart (your current winning setup)

**Frozen LLM + HRM, CAB (multi-token), vocab-bias head, 2×2 train / 4-segment eval:**

```bash
python3 train.py \
  --model_name ./tiny-rl-sft \
  --tasks basic_arithmetic,gsm_symbolic,chain_sum,simple_equations \
  --segments 2 --inner_T 2 \
  --injector cab \
  --batch_size 2 --lr 3e-5 \
  --max_steps 200000 \
  --save_every 10000 \
  --eval_every 500 --eval_n 200 --eval_seed_jitter \
  --log_samples_every 500 \
  --max_new_tokens 512 \
  --grad_clip 1.0 \
  --eval_segments 4
```

> Tip: For more steering, try `--eval_segments 6`.
> If you added CLI wiring for gates/memory (see below), also try:
> `--cab_mem 4 --cab_gate_init -1.0 --vocab_gate_init -1.0`.

---

## What Trains (and What Stays Frozen)

* **Frozen**: the base LLM (all transformer layers, LM head weights).

  * We explicitly set `p.requires_grad_(False)` on the LLM and run `forward_llm_hidden` under `torch.no_grad()`.
* **Trainable**:

  * HRM: `HBlock`, `LBlock`, `RMSNorm`, learnable initial states `zH0`, `zL0`, and the projection layers (`pool_mix`, `x_proj`).
  * Injector: **GRB** or **CAB** (now with multi-token memory).
  * Optional **vocab-bias head** (and its gate).
  * Optional **q\_head** if ACT is enabled.

---

## CLI Flags of Interest

Base (already in `train.py`):

* `--segments`, `--inner_T` – HRM depth (train).
* `--eval_segments` – HRM depth at eval (inference scaling).
* `--injector {grb,cab}` – choose the bridge.
* `--temperature` – CE logit temperature for training (e.g., `0.7` for sharper CE).
* `--batch_size`, `--lr`, `--max_steps`, `--save_every`, `--eval_every`, `--eval_n`, `--grad_clip`…

Recommended extras (add if not present yet):

* `--cab_mem` (int, default 4): CAB memory tokens.
* `--cab_gate_init` (float, default -1.5): initial CAB gate (sigmoid).
* `--grb_gate_init` (float, default -2.0): initial GRB gate.
* `--vocab_gate_init` (float, default -2.0): initial logit-bias gate.

Wire these into `HRMConfig` and the injector/vocab-bias init so you can sweep from the CLI.

---

## Results Snapshot (example)

After wiring **multi-token CAB** and **vocab-bias head** (and ensuring eval calls `logits_from_injected(..., zH=zH)`), we’ve seen:

```
[EVAL 50000] acc_proxy ≈ 0.95 on n=200 (greedy decode)
```

Greedy decoding (T=1.0) on arithmetic and short word-math mixes returns clean, numeric final answers with EOS termination.

---

## Training & Eval Flow

1. **Frozen LLM forward** to get last hidden states for (prompt + target).
2. **Pool prompt** only: mix of (mean + last), then project to HRM width.
3. **HRM recurrence**: `segments × inner_T`, with detach between segments.
4. **Inject** `z_H` into LLM final hidden states via **GRB** or **CAB**.
5. **Final norm (if present)** → **LM head** → **logits**;
   add **vocab-bias** from `z_H` when available.
6. **Loss**: Cross-entropy over **target region** (prompt tokens masked with `-100`).
7. **Eval/spot-check**: greedy decode over target region (and make sure the **vocab-bias** is active by passing `zH`).

---

## Checkpoints & Compatibility

* Changing from **1-token → multi-token CAB** or adding the **vocab-bias head** means **new tensors**.

  * **Best**: start fresh.
  * **Warm-start**: implement a **partial loader** that loads shared pieces and skips missing ones; re-init new parts; start with a fresh optimizer.
* Old checkpoints from the single-token CAB won’t load into the multi-token CAB unless you skip shape-mismatched keys.

---

## Troubleshooting

* **Blank / EOS-only predictions at eval**
  Ensure eval uses the vocab-bias head:

  ```python
  logits = model.logits_from_injected(model.injector(last_hidden, zH), zH=zH)
  ```
* **Steering feels too weak**
  Increase HRM depth at eval (`--eval_segments 6`), bump CAB gate (e.g., `--cab_gate_init -1.0`), or increase `mem_tokens` to 4 or 8.
* **Spiky loss early on**
  Use `--grad_clip 1.0`, try LR in `3e-5 … 1e-4`, and `--temperature 0.7` (training CE only).
* **Checkpoint fails to load**
  Use the partial loader (skip missing keys) or start a new run.

---

## Example Commands

**Your current run (frozen LLM, CAB, 2×2 train / 4 eval):**

```bash
python3 train.py \
  --model_name ./tiny-rl-sft \
  --tasks basic_arithmetic,gsm_symbolic,chain_sum,simple_equations \
  --segments 2 --inner_T 2 \
  --injector cab \
  --batch_size 2 --lr 3e-5 \
  --max_steps 200000 \
  --save_every 10000 \
  --eval_every 500 --eval_n 200 --eval_seed_jitter \
  --log_samples_every 500 \
  --max_new_tokens 512 \
  --grad_clip 1.0 \
  --eval_segments 4
```

**Ablation: deeper eval, stronger gates**

```bash
python3 train.py \
  --model_name ./tiny-rl-sft \
  --tasks basic_arithmetic,gsm_symbolic,chain_sum,simple_equations \
  --segments 2 --inner_T 2 \
  --injector cab \
  --cab_mem 8 --cab_gate_init -1.0 \
  --vocab_gate_init -1.0 \
  --batch_size 2 --lr 5e-5 --temperature 0.7 \
  --max_steps 200000 --save_every 10000 \
  --eval_every 500 --eval_n 200 \
  --eval_segments 6 \
  --log_samples_every 500 --max_new_tokens 512 \
  --grad_clip 1.0
```

---

## Limitations & Next Steps

* **Frozen LLM**: great stability, but caps ceiling; consider optional small LoRA or partial unfreezing (e.g., final norm + lm\_head) once hybrid stabilizes.
* **Task breadth**: extend beyond arithmetic to multi-step word problems, symbolic algebra, small logic tasks.
* **Scheduling**: optional warmup / cosine decay can help; currently fixed LR is fine.

Planned:

* CLI control for CAB memory and gates (if not already added).
* “Best on eval” checkpointing & richer eval metrics (sign errors, numeric noise, per-length accuracy).
* Optional static numeric bias on target positions as a safeguard (usually unnecessary now).

---

## References

* *Hierarchical Reasoning Model* (Sapient Intelligence, 2025). arXiv:2506.21734
* Sapient HRM GitHub: [https://github.com/sapientinc/HRM](https://github.com/sapientinc/HRM)
* Lucidrains HRM (PyTorch): [https://github.com/lucidrains/HRM](https://github.com/lucidrains/HRM)

---

If you want, I can also make a PR-ready `README.md` with the CLI flags actually present in your current `train.py` (or include diffs to add them).

