
---

# HRM-LLM Hybrid

## Overview

This repository implements a **Hierarchical Reasoning Model (HRM)** controller on top of a **frozen language model (LLM)**. The LLM provides fluent text generation, while HRM contributes structured latent reasoning in hidden space.

Unlike standard Chain-of-Thought prompting (which externalizes reasoning into text), HRM enables the system to **“think silently” in hidden states** before producing an answer. This yields deeper, more reliable reasoning without long token chains.

Key ingredients:

* **Frozen Hugging Face LLM** (TinyLlama, Qwen, etc.)
* **Hierarchical Reasoning Model (HRM)**: coupled high-level (H) and low-level (L) recurrent modules.
* **Injectors** (GRB or CAB) to bias LLM hidden states with HRM’s latent reasoning output.
* **Optional ACT halting** (q-head) to adapt reasoning depth per query.
* **Reasoning-Gym datasets** for arithmetic, equations, and symbolic tasks.
* **Minimal training harness** with deep supervision, spot-check logging, and inference-time scaling.

---

## Quick Results

Our first stable training run (≈ **50,000 steps**) with a frozen **TinyLlama** backbone + **HRM controller** reached:

* **Proxy Eval Accuracy:** \~**95%** (n=200, greedy decode)
* **Tasks:** arithmetic (addition, subtraction, equations, GSM symbolic mix)
* **z\_H mean norm:** \~13
* **Training Loss:** fell to near-zero on supervised segments
* **Spot-check outputs:** exact numeric answers, EOS termination


### Current HRMConfig

```python
class HRMConfig:
    # Latent width / depth
    d_h: int = 512
    h_layers: int = 4
    l_layers: int = 4
    n_heads: int = 8

    # Unrolled dynamics
    inner_T: int = 3
    segments: int = 3

    # Bridges / options
    use_cab: bool = False     # False → GRB, True → CAB
    use_act: bool = False

    # Small vocab-bias head (z_H → logits bias), gated
    logit_bias_head: bool = True
    logit_bias_init: float = -2.0  # sigmoid(-2) ≈ 0.12 initial strength

    # Label masking for CE
    vocab_ignore_index: int = -100
```

### Approximate Model Size

* **Base LLM (frozen):** TinyLlama-1.1B (\~1.1B params)
* **HRM controller + injectors + bias head:** \~**25–30M params** (depending on hidden size and CAB mem tokens)
* **Total trainable parameters:** \~**30M** (≈ 3% of backbone)

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

## Motivation

Current LLMs excel at language but struggle with robust, multi-step reasoning. Chain-of-Thought (CoT) prompting helps, but it is:

* **Brittle**: a single token error can derail the whole chain.
* **Data-hungry**: requires many demonstrations.
* **Slow**: long token chains inflate latency and cost.

HRM provides **latent recurrence** in hidden space instead of tokens:

* **Two timescales** of reasoning:

  * **L-module** (fast, detailed, multiple micro-steps per cycle).
  * **H-module** (slow, abstract, 1 step per cycle).
* **Hierarchical convergence**: the L-module converges locally, then resets under updated guidance from the H-module.
* **O(1) memory training** via one-step gradient + deep supervision.
* **ACT halting**: decide dynamically how many reasoning cycles to run.
* **Inference-time scaling**: increase segments at inference to “think longer” without retraining.

---

## Architecture Overview

The pipeline combines a frozen LLM with HRM reasoning and injects the results back into the LLM hidden space before decoding.

👉 *(Insert the ASCII diagram we built together here — showing Input → Frozen LLM → Pool → HRM Loop → Injector → Norm → LM Head → Output, with ACT/q-head and vocab bias.)*

---

### Components

#### 1. Frozen LLM

* Provides embeddings, hidden states, and LM head.
* Parameters frozen (no fine-tuning required).
* Final normalization layer optionally applied before logits.

#### 2. Pooling & Projection

* Mean+last pooling over **prompt tokens only**.
* Linear projection to HRM dimension (`d_h`, default 512).

#### 3. HRM Core

* **L-block**: runs multiple fast inner steps per segment.
* **H-block**: runs once per segment, steering the L-block across cycles.
* **Hierarchical convergence**: keeps computation active across segments.
* **RMSNorm** applied to z\_H.
* **Deep supervision**: loss applied at each segment, with hidden states detached in between.

#### 4. Injectors

* **GRB (Gated Residual Bias)**: projects z\_H as a bias vector, gated by a learnable sigmoid scalar.
* **CAB (Cross-Attention Bridge)**: treats z\_H as one or more memory tokens; token hidden states attend to it. Multi-token CAB offers stronger conditioning.
* Both support residual gating to prevent destabilization.

#### 5. Optional Heads

* **ACT / q-head**: predicts halt vs. continue from z\_H at each segment. Enables Adaptive Computation Time.
* **Vocab bias head**: adds a gated bias over vocabulary logits from z\_H. Helpful for structured outputs like numbers.

#### 6. Training Procedure

* Cross-entropy loss on target tokens (prompt region masked).
* Optional q-loss for ACT halting.
* Segment-level deep supervision with detach → O(1) memory.
* AdamW optimizer with weight decay.
* Gradient clipping for stability.

---

## Datasets

* **Toy addition / chain sum**
* **Basic arithmetic**
* **Simple equations**
* **GSM symbolic reasoning**

Wrapped with a collator that:

* Tokenizes prompt and target separately.
* Concatenates them with masking (`ignore_index=-100` for prompt region).
* Returns `input_ids`, `attention_mask`, `labels`, `prompt_lengths`, and raw examples (for verifiers).

---

## Training Harness

The provided `train.py` script supports:

* **Arguments**:

  * `--segments`, `--inner_T`: HRM loop depth.
  * `--injector {grb,cab}`: choose injector.
  * `--use_act`, `--act_penalty`: enable ACT/Q-head.
  * `--decode_temperature`, `--decode_top_p`, `--decode_top_k`: sampling options at eval.
  * `--eval_segments`: run deeper reasoning at eval-time.
  * `--grad_clip`: stabilize updates.

* **Logging**:

  * Per-step losses.
  * Per-segment losses and q-losses.
  * Proxy eval accuracy with verifier functions.
  * Spot-check decoded examples with greedy or sampling decode.
  * CSV logging of eval curves.

* **Checkpoints**:

  * Lightweight save of trainable HRM parts + optimizer state.
  * Resume training with `--resume path/to/checkpoint.pt`.

---

## Example Usage

**Arithmetic reasoning (TinyLlama, CAB injector, 2×2 loop):**

```bash
python3 train.py \
  --model_name ./tiny-rl-sft \
  --tasks basic_arithmetic,gsm_symbolic,chain_sum,simple_equations \
  --segments 2 --inner_T 2 \
  --injector cab \
  --batch_size 2 --lr 1e-5 \
  --max_steps 200000 \
  --save_every 10000 \
  --eval_every 500 --eval_n 200 --eval_segments 4 \
  --log_samples_every 500 --max_new_tokens 512
```

---

## Key Properties

✅ **Latent reasoning**: computation happens in hidden space, not tokens.
✅ **Constant memory**: deep supervision + one-step gradient avoid BPTT.
✅ **Inference-time scaling**: simply raise `segments` at eval to think deeper.
✅ **Optional ACT halting**: saves compute by stopping early.
✅ **Flexible injectors**: GRB (lightweight) vs CAB (stronger conditioning).
✅ **Frozen LLM**: no costly full-model fine-tuning.

---

## Limitations & Next Steps

⚠️ **Base LLM frozen**: limits synergy; adding LoRA to upper layers may help.
⚠️ **Stablemax**: currently approximated with temperature scaling; future work could implement true stablemax.
⚠️ **Task coverage**: focused on math/symbolic; needs broader datasets.
⚠️ **ACT stability**: q-head halting works but can be finicky; reinforcement-style training could improve.

**Planned extensions:**

* LoRA adapters for partial LLM tuning.
* YAML/JSON configs for reproducible runs.
* Integration with Cogito memory modules (episodic/graph memory).
* Experiments with HRM-only (no LLM) baselines.

---

## References

* Sapient Intelligence. *Hierarchical Reasoning Model*. 2025. [arXiv:2506.21734](https://arxiv.org/abs/2506.21734)&#x20;
* Sapient Intelligence HRM GitHub: [github.com/sapientinc/HRM](https://github.com/sapientinc/HRM)
* Lucidrains experimental HRM repo: [github.com/lucidrains/HRM](https://github.com/lucidrains/HRM)

---


