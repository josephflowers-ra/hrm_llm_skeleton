
---

# HRM-LLM Skeleton

## Overview

This repository explores augmenting small, open-weight language models (TinyLlama, Qwen2.5, etc.) with the **Hierarchical Reasoning Model (HRM)** — a biologically inspired recurrent architecture designed for deep, efficient reasoning.

The motivation: while Transformers are powerful at language generation, their reasoning tends to be shallow and token-bound (via Chain-of-Thought prompting). HRM introduces **latent reasoning in hidden space** at two timescales, mirroring how the brain separates fast and slow cognitive processes. By combining HRM with a frozen LLM, we create a hybrid system that can **think silently in hidden states** while the LLM remains the **fluent communicator**.

This project provides a clean skeleton for experimentation:

* HRM blocks (fast + slow recurrence, hierarchical convergence).
* Feature bridge from LLM → HRM → LLM.
* Injectors to bias LLM outputs with latent states.
* Deep supervision + one-step gradients (O(1) memory, no BPTT).
* Optional ACT halting, guided by verifier signals.
* Reasoning-Gym adapter with toy tasks for quick testing.
* Minimal training harness with Hugging Face integration.

---

## Background: What is HRM?

The **Hierarchical Reasoning Model (HRM)** was introduced by Sapient Intelligence (2025) as a recurrent neural architecture that performs reasoning in a latent space, without producing long token-based chains.

**Core principles:**

* **Two timescales of computation**

  * **L-module (low-level, fast):** Executes multiple rapid micro-steps per segment, converging to local solutions.
  * **H-module (high-level, slow):** Updates once per segment, guiding the L-module across cycles.
* **Hierarchical convergence:** The L-module repeatedly converges within each cycle, while the H-module resets and steers it toward a new local equilibrium.
* **One-step gradient approximation:** Training uses the final step only (no Backpropagation Through Time), reducing memory from O(T) → O(1).
* **Deep supervision:** Loss is applied at the end of every segment, with hidden states detached between them.
* **Adaptive Computation Time (ACT):** A halting head decides whether to stop or continue reasoning, balancing accuracy and compute.

**Why it matters:**

* **Data efficiency:** HRM solves tasks like Sudoku-Extreme and Maze-Hard with \~1k examples — where LLMs with CoT fail completely.
* **Compute efficiency:** Latent reasoning is compact; no long CoT token chains.
* **Scalability:** Increase segment count at inference to boost depth without retraining.
* **Reliability:** Silent reasoning avoids brittle token-level step errors.

---

## How This Repo Extends Small LLMs

We graft HRM onto a frozen Hugging Face LLM. The LLM provides **fluent language outputs**, while HRM supplies **structured latent reasoning**.

### Dataflow

1. **Input:** Prompt tokens → frozen LLM → last hidden states.
2. **Pooling:** Mean-pool over prompt tokens → compact embedding (`x̃`).
3. **HRM loop:**

   * For each segment *m*:

     * Run (T−1) L-steps (no\_grad).
     * Run final L-step (grad) + H-step (grad).
     * z\_H^m is detached before next segment.
4. **Injection:** z\_H^m biases the LLM’s final hidden states via injector (GRB or CAB).
5. **Output:** Conditioned hidden states → LM head → token predictions.
6. **Loss:** Cross-entropy on target tokens, plus optional ACT/Q-head loss.

---

## Repository Components

### 1. HRM Blocks (`hrm_blocks.py`)

* **HBlock / LBlock:** small encoder-only Transformer stacks.
* **RMSNorm, SwiGLU MLP, bias-free linear layers** for stability.
* Dimensions default to `d=512`, `n_heads=8`, expansion=4 — matching the HRM small config.

### 2. Injectors (`injector.py`)

* **InjectorGRB (default):** gated residual bias (simple, strong).
* **CrossAttentionBridge (CAB):** cross-attn from tokens to z\_H memory token (soft conditioning).
* Toggle via `HRMConfig(use_cab=True)`.

### 3. Controller (`model.py`)

* Loads frozen Hugging Face LLM.
* Pools prompt tokens, projects to HRM dimension.
* Runs HRM segments (`segments × inner_T`).
* Injects z\_H into LLM hidden states before LM head.
* Implements training with:

  * Cross-entropy loss.
  * Deep supervision per segment.
  * Optional ACT/Q-head loss.

### 4. Datasets (`reasoning_gym_wrapper.py`)

* **ToyAddDataset:** simple integer addition with programmatic verifier.
* **Reasoning-Gym adapter:** plug in tasks like `arithmetic`, `maze`.

### 5. Training Harness (`train.py`)

* CLI arguments:

  * `--segments`, `--inner_T`, `--use_act`, `--act_penalty`, `--temperature`.
  * `--task {toy_add, arithmetic, maze, ...}`.
* Uses **AdamW** optimizer (betas 0.9/0.95, wd=0.1).
* Logs segment losses, Q-losses, and proxy eval accuracy.

### 6. Dependencies (`requirements.txt`)

```
torch
transformers
datasets
accelerate
einops
reasoning-gym
```

---

## Example Usage

**Toy addition (TinyLlama, GRB injector, 3×3 loop):**

```bash
python3 train.py \
  --model_name TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --task toy_add \
  --segments 3 --inner_T 3 \
  --batch_size 4 --lr 1e-4 --max_steps 200
```

**Arithmetic with ACT halting (Qwen2.5, CAB injector, ACT penalty):**

```bash
python3 train.py \
  --model_name Qwen/Qwen2.5-1.5B \
  --task arithmetic \
  --segments 4 --inner_T 4 \
  --use_act --act_penalty 0.001 \
  --batch_size 2 --lr 1e-4 --max_steps 1000
```

**Maze navigation (TinyLlama, deeper reasoning):**

```bash
python3 train.py \
  --model_name TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --task maze \
  --segments 6 --inner_T 4 \
  --batch_size 2 --lr 5e-5 --max_steps 2000
```

---

## What This Project Adds

✅ HRM latent controller (H/L recurrence, hierarchical convergence).
✅ Injectors to bias frozen LLM hidden states.
✅ Deep supervision + one-step gradient (constant memory).
✅ Optional verifier-driven ACT halting.
✅ Minimal training harness + reasoning-gym integration.

---

## Limitations & Next Steps

⚠️ **Stablemax:** Currently approximated by logit temperature scaling; a true stablemax head could be added.
⚠️ **ACT training:** Uses supervised halting signals; future work could add exploration (ε-greedy).
⚠️ **Frozen LLM:** Base LLM is fixed. Optionally, LoRA adapters can be added to top layers for synergy.
⚠️ **Task scope:** Currently supports reasoning-gym and toy addition; needs broader datasets for richer evaluation.

**Planned extensions:**

* Add YAML config parser (mirror HRM model cards).
* LoRA flag for fine-tuning top LLM layers.
* Integration with Cogito memory modules (episodic/graph memory + HRM).
* Ablations: GRB vs CAB, scaling segments/inner\_T, with/without ACT.

---

## References

* Guan Wang, Jin Li, Yuhao Sun, Xing Chen, Changling Liu, Yue Wu, Meng Lu, Sen Song, Yasin Abbasi Yadkori.
  *Hierarchical Reasoning Model*. Sapient Intelligence (2025).
  [arXiv:2506.21734](https://arxiv.org/abs/2506.21734)&#x20;
* [Sapient Intelligence HRM GitHub](https://github.com/sapientinc/HRM)

---

