# HRM + LLM Skeleton (TinyLlama/Qwen2.5) + Reasoning-Gym

A minimal, *practical* scaffold to add **Hierarchical Reasoning Model (HRM)**-style latent reasoning
to a small, permissively-licensed LLM (e.g., TinyLlama 1.1B, Qwen2.5-1.5B) and train on
**reasoning-gym** tasks with **deep supervision** and **1-step gradient** (no BPTT).

> ✅ Default: LLM frozen. Train HRM blocks + injector only.
> ✅ Deep supervision across segments with hidden-state detach.
> ✅ Optional ACT (halting) head stub included.
> ✅ Works with HuggingFace `transformers` and `reasoning-gym` if installed.

---

## Quick Start

```bash
# 1) Create venv and install deps
python3 -m venv .venv && source .venv/bin/activate
python3 -m pip install --upgrade pip

# You can start with CPU; enable CUDA if available
python3 -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python3 -m pip install transformers datasets accelerate einops

# Optional (recommended) - Reasoning-Gym
python3 -m pip install reasoning-gym

# 2) Run a tiny smoke test (toy dataset if reasoning-gym not installed)
python3 train.py --model_name TinyLlama/TinyLlama-1.1B-Chat-v1.0                  --segments 3 --inner_T 3 --lr 2e-4 --batch_size 2 --max_steps 50
```

> Replace `--model_name` with `Qwen/Qwen2.5-1.5B` if you prefer. Both are Apache-2.0.

---

## Design

- **HRM blocks**: `HBlock` (slow) + `LBlock` (fast), lightweight encoder-only transformer layers.
- **Hierarchical convergence**: For each segment, L runs `T` fast steps; H updates once and resets L.
- **Deep supervision**: After each segment, compute loss → update → **detach** state → next segment.
- **1-step gradient**: Only backprop through the *final* (H,L) updates for each segment.
- **Injector** (choose one):
  - `InjectorGRB`: *Gated Residual Bias* — adds a broadcasted bias derived from z_H to last hidden states.
  - `InjectorCAB`: *Cross-Attention Bridge* — single cross-attn from tokens to a learned z_H token.
- **Frozen LLM**: We forward the LLM with `torch.no_grad()` to get token hidden states, then inject.
  Gradients flow into HRM+Injector through the LM head operation while LLM weights stay frozen.

---

## Files

- `hrm_blocks.py` — H/L transformer blocks (RMSNorm, GLU MLP).
- `injector.py` — InjectorGRB (default) and InjectorCAB.
- `model.py` — `HRMController` that wraps HF LLM, runs segments, computes losses.
- `reasoning_gym_wrapper.py` — dataset adapters; falls back to a toy arithmetic dataset if gym missing.
- `train.py` — training loop with deep supervision + 1-step gradient; metrics & prints.
- `requirements.txt` — minimal deps list.

---

## Notes

- **Teacher forcing**: We use standard causal language modeling loss on the target tokens (CE).
- **Verifier loss**: If the task provides a programmatic verifier, we add a small auxiliary loss.
- **ACT (halting)**: A Q-head stub is provided; by default we use fixed `segments`. Flip `--use_act` to try it.
- **LoRA (optional)**: If you need more capacity, add LoRA to the top N LLM layers. Not included by default.

---

## Example Command Lines

```bash
# TinyLlama, 3 segments, inner T=3, GRB injector
python3 train.py --model_name TinyLlama/TinyLlama-1.1B-Chat-v1.0   --segments 3 --inner_T 3 --batch_size 4 --lr 1e-4 --max_steps 1000

# Qwen2.5-1.5B, 4 segments, inner T=4, with ACT enabled
python3 train.py --model_name Qwen/Qwen2.5-1.5B   --segments 4 --inner_T 4 --use_act   --batch_size 2 --lr 1e-4 --max_steps 1000
```

---

## License

This scaffold is provided under the **Apache-2.0** license. Check the licenses of the base models you use:
- TinyLlama: Apache-2.0
- Qwen2.5 (1.5B): Apache-2.0
