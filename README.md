
---

# HRM + LLM Skeleton

## Overview

This project explores augmenting **small, permissively licensed language models** (e.g., TinyLlama, Qwen2.5) with the **Hierarchical Reasoning Model (HRM)**. The goal is to combine the **language fluency** of Transformers with the **structured latent reasoning** of HRM.

Instead of relying only on token-level **Chain-of-Thought**, which can be brittle and inefficient, this system performs **latent reasoning** inside HRM modules and then conditions the base LLM to produce the final answer.

---

## What is HRM?

The **Hierarchical Reasoning Model (HRM)** is a brain-inspired recurrent architecture introduced by Sapient Intelligence (2025). It addresses limitations of standard Transformers and RNNs in reasoning:

* **Two timescales of computation**

  * **L-module**: fast, low-level executor that runs multiple micro-steps.
  * **H-module**: slower, high-level planner that updates once per cycle and guides the L-module.

* **Hierarchical Convergence**
  The L-module converges locally, then the H-module resets it for the next stage. This prevents premature convergence and enables deep, multi-stage reasoning.

* **One-Step Gradient Approximation**
  HRM avoids Backpropagation Through Time (BPTT). Instead, it computes gradients only through the final step of each segment, keeping memory O(1) and training stable.

* **Deep Supervision**
  Losses are applied at the end of every segment, with hidden states detached between segments. This stabilizes training and improves generalization.

* **Adaptive Computation Time (ACT)**
  An optional halting head learns whether to stop or continue reasoning, saving compute on easy problems and thinking longer on hard ones.

HRM achieves **near-perfect results** on Sudoku-Extreme and Maze-Hard with only \~1k examples, outperforming larger CoT-based models.

---

## How the Hybrid HRM + LLM Works

1. **Frozen LLM**
   A base LLM (TinyLlama, Qwen2.5, etc.) provides language understanding and fluency. Its hidden states are extracted but not fine-tuned.

2. **Latent Reasoning (HRM loop)**

   * Mean-pooled prompt embeddings are projected into HRM space.
   * HRM runs for multiple *segments* (`M`), with each segment containing several *inner steps* (`T`).
   * States are detached between segments (deep supervision).

3. **Injectors (HRM → LLM)**
   HRM’s high-level state `z_H` conditions the LLM’s hidden states via one of two adapters:

   * **Gated Residual Bias (GRB):** add a small bias vector to all token states.
   * **Cross-Attention Bridge (CAB):** attend over a projected `z_H` as a memory key/value.

4. **Language Output**
   Conditioned hidden states are passed through the LLM’s output head to generate tokens.

5. **Training**

   * **Cross-entropy loss** on targets (with optional logit temperature scaling).
   * **ACT loss (optional):** trains a Q-head to predict halting, using dataset verifiers for correctness.
   * Optimizer: AdamW with betas (0.9, 0.95) and weight decay (0.1).

---

## Data & Verification

* **Reasoning-Gym Integration**: tasks like arithmetic, mazes, logic.
* **Toy Addition Dataset**: default fallback with built-in `verify()` function.
* Verifiers are critical: they provide correctness signals for **ACT halting**.

---

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run training (toy addition by default)
python3 train.py --model_name TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --task toy_add --segments 3 --inner_T 3 --batch_size 4 \
  --lr 1e-4 --max_steps 200

# Run with ACT halting
python3 train.py --model_name Qwen/Qwen2.5-1.5B \
  --task arithmetic --segments 4 --inner_T 4 \
  --use_act --act_penalty 0.001 --batch_size 2 --max_steps 500
```

---

## Repo Structure

* `hrm_blocks.py` – defines **HBlock**, **LBlock**, Transformer components (RMSNorm, SwiGLU, attention).
* `injector.py` – injectors (GRB and CAB).
* `model.py` – `HRMController`: HRM loop, injectors, training step.
* `reasoning_gym_wrapper.py` – dataset adapters + toy dataset.
* `train.py` – CLI training harness with eval, logging, optimizer.

---

## Next Steps

* Add **stablemax loss** (used in HRM paper).
* Extend **ACT halting** with exploration strategies.
* Optional **LoRA adapters** to train top LLM layers.
* Curriculum across Reasoning-Gym tasks.


