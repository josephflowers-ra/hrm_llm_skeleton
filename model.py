#!/usr/bin/env python3
import torch
import torch.nn as nn
from dataclasses import dataclass
from transformers import AutoModelForCausalLM, AutoTokenizer

from hrm_blocks import HBlock, LBlock, RMSNorm
from injector import InjectorGRB, CrossAttentionBridge


@dataclass
class HRMConfig:
    """
    Configuration for the HRM controller and its interface to the frozen LLM.
    """
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


class HRMController(nn.Module):
    """
    HRM + LLM hybrid controller (LLM is frozen).

    - Frozen HF CausalLM provides the language backbone ("speaker").
    - HRM (H/L blocks) runs fast/slow latent recurrence on pooled prompt features.
    - Injector (GRB or CAB) conditions the LLM hidden states with the final z_H.
    - Optional vocab-bias head adds a tiny, gated bias from z_H directly to logits.
    - Only HRM + injector + projections (+ bias head) are trainable.

    Typical dataflow (per step):
      hidden = LLM(prompt+target) [frozen, w/ hidden states]
      x_pool  = mixed(mean+last of prompt hidden)
      z_H     = HRM(x_pool; inner_T, segments)
      hidden' = Inject(hidden, z_H)
      logits  = LMHead(hidden'); optionally add gated vocab bias from z_H
      loss    = CE(logits, labels)  [labels mask prompt tokens with -100]
    """
    def __init__(self, model_name: str, hrm_cfg: HRMConfig):
        super().__init__()
        self.hrm_cfg = hrm_cfg

        # --- Tokenizer ---
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        # Right padding is typically best for CLM teacher-forcing batches
        if getattr(self.tokenizer, "padding_side", "right") != "right":
            self.tokenizer.padding_side = "right"
        if self.tokenizer.pad_token is None:
            # Ensure a pad token exists; many causal LMs reuse EOS
            self.tokenizer.pad_token = self.tokenizer.eos_token
        if not hasattr(self.tokenizer, "eos_token_id") or self.tokenizer.eos_token_id is None:
            # Very rare, but make sure EOS exists as well
            self.tokenizer.eos_token = self.tokenizer.pad_token

        # --- Frozen LLM ---
        self.lm = AutoModelForCausalLM.from_pretrained(model_name)
        for p in self.lm.parameters():
            p.requires_grad_(False)
        self.lm.eval()

        d_model = self.lm.config.hidden_size
        V = self.lm.get_output_embeddings().weight.size(0)

        # --- HRM blocks (trainable) ---
        self.h_block = HBlock(d=self.hrm_cfg.d_h, n_layers=self.hrm_cfg.h_layers, n_heads=self.hrm_cfg.n_heads)
        self.l_block = LBlock(d=self.hrm_cfg.d_h, n_layers=self.hrm_cfg.l_layers, n_heads=self.hrm_cfg.n_heads)
        self.in_norm = RMSNorm(self.hrm_cfg.d_h)

        # Prompt pooling: (mean + last) -> mix -> project to HRM width
        self.pool_mix = nn.Linear(d_model * 2, d_model, bias=False)
        self.x_proj = nn.Linear(d_model, self.hrm_cfg.d_h, bias=False)

        # Learnable initial states (expanded per batch)
        self.zH0 = nn.Parameter(torch.zeros(1, 1, self.hrm_cfg.d_h))
        self.zL0 = nn.Parameter(torch.zeros(1, 1, self.hrm_cfg.d_h))

        # --- Injector: GRB or CAB (trainable) ---
        if self.hrm_cfg.use_cab:
            self.injector = CrossAttentionBridge(
                d_h=self.hrm_cfg.d_h, d_model=d_model, n_heads=self.hrm_cfg.n_heads
            )
        else:
            self.injector = InjectorGRB(d_h=self.hrm_cfg.d_h, d_model=d_model)

        # Optional ACT (halt) head on zH
        self.q_head = nn.Linear(self.hrm_cfg.d_h, 1) if self.hrm_cfg.use_act else None

        # Optional tiny vocab-bias head (trainable)
        if self.hrm_cfg.logit_bias_head:
            self.vocab_bias = nn.Linear(self.hrm_cfg.d_h, V, bias=False)
            self.vocab_gate = nn.Parameter(torch.tensor(self.hrm_cfg.logit_bias_init))
        else:
            self.vocab_bias = None
            self.vocab_gate = None

    # -----------------------
    #  Convenience
    # -----------------------
    def set_segments(self, segments: int):
        """Set default number of HRM segments (e.g., deeper reasoning at inference)."""
        self.hrm_cfg.segments = int(segments)

    # -----------------------
    #  LLM helper methods
    # -----------------------
    def forward_llm_hidden(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Forward the frozen LLM and return last hidden states (B, T, D).
        Explicitly disable cache and grads; we only need hidden states.
        """
        with torch.no_grad():
            out = self.lm(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                use_cache=False,
            )
            return out.hidden_states[-1]

    def _apply_final_norm(self, hidden: torch.Tensor) -> torch.Tensor:
        """
        Some architectures (LLaMA/Mistral-like, NeoX) expect a final LayerNorm
        before the LM head. Apply it when present; otherwise, identity.
        """
        # LLaMA/Mistral family: lm.model.norm
        try:
            norm = getattr(getattr(self.lm, "model", None), "norm", None)
            if norm is not None:
                return norm(hidden)
        except Exception:
            pass
        # GPT-NeoX family: lm.gpt_neox.final_layer_norm
        try:
            neox = getattr(self.lm, "gpt_neox", None)
            if neox is not None and hasattr(neox, "final_layer_norm"):
                return neox.final_layer_norm(hidden)
        except Exception:
            pass
        # Default: no-op
        return hidden

    def logits_from_injected(self, injected_hidden: torch.Tensor, zH: torch.Tensor | None = None) -> torch.Tensor:
        """
        Apply final norm (if applicable) then LM head → logits (B, T, V).
        If zH is provided and a vocab-bias head is enabled, add a gated bias over vocabulary.
        """
        hidden = self._apply_final_norm(injected_hidden)
        logits = self.lm.get_output_embeddings()(hidden)
        if (zH is not None) and (self.vocab_bias is not None):
            # zH: (B,1,Dh) → bias (B,V) → add to every timestep (target masking is handled by CE)
            bias = self.vocab_bias(zH).squeeze(1)  # (B,V)
            g = torch.sigmoid(self.vocab_gate)     # scalar in (0,1)
            logits = logits + g * bias.unsqueeze(1)
        return logits

    # -----------------------
    #  HRM dataflow
    # -----------------------
    def pool_tokens(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor, prompt_lengths) -> torch.Tensor:
        """
        Mixed pooling over ONLY the prompt region per sample → (B, 1, D_model).
          pooled = mix( mean(prompt), last(prompt) )
        """
        pooled = []
        T = hidden_states.size(1)
        for i, Lp in enumerate(prompt_lengths):
            Lp_i = max(1, min(int(Lp), T))
            mean_p = hidden_states[i, :Lp_i, :].mean(0)
            last_p = hidden_states[i, Lp_i - 1, :]
            mixed = self.pool_mix(torch.cat([mean_p, last_p], dim=-1))
            pooled.append(mixed)
        return torch.stack(pooled, 0).unsqueeze(1)  # (B,1,D_model)

    def hrm_segments(self, x_tilde: torch.Tensor, segments: int, inner_T: int):
        """
        Run HRM for `segments` cycles.
        - First inner_T-1 L-steps under no_grad (let L converge)
        - Final L-step + one H-step with grad
        - Deep supervision: detach(zH,zL) between segments  → O(1) memory
        Returns list of zH per segment (len == segments).
        """
        B = x_tilde.size(0)
        zH = self.zH0.expand(B, -1, -1)
        zL = self.zL0.expand(B, -1, -1)
        zH_list = []

        for _ in range(segments):
            with torch.no_grad():
                zH_t, zL_t = zH, zL
                for _ in range(max(0, inner_T - 1)):
                    zL_t = self.l_block(zL_t, zH_t, x_tilde)

            # Final grad-carrying updates
            zL = self.l_block(zL_t, zH_t, x_tilde)
            zH = self.h_block(zH_t, zL)
            zH = self.in_norm(zH)

            zH_list.append(zH)
            # Detach between segments for O(1) memory
            zH = zH.detach()
            zL = zL.detach()

        return zH_list

    # -----------------------
    #  Training step
    # -----------------------
    def training_step(
        self,
        batch: dict,
        segments: int = None,
        inner_T: int = None,
        use_act: bool = False,
        temperature: float = 1.0,
        act_penalty: float = 0.0,
    ):
        """
        One step of training with deep supervision across segments.
        Returns (total_loss, metrics) where metrics includes per-segment CE,
        optional q_losses (if ACT), and mean ||z_H|| for quick health checks.
        """
        segments = segments or self.hrm_cfg.segments
        inner_T = inner_T or self.hrm_cfg.inner_T

        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]
        prompt_lengths = batch["prompt_lengths"]

        # 1) Frozen LLM forward to get hidden states
        last_hidden = self.forward_llm_hidden(input_ids, attention_mask)

        # 2) Pool prompt → mix → project to HRM width
        x_pool = self.pool_tokens(last_hidden, attention_mask, prompt_lengths)
        x_tilde = self.x_proj(x_pool)

        # 3) HRM loop
        zH_list = self.hrm_segments(x_tilde, segments=segments, inner_T=inner_T)

        ce = nn.CrossEntropyLoss(ignore_index=self.hrm_cfg.vocab_ignore_index)
        metrics = {"segment_losses": [], "q_losses": []}
        total_loss = 0.0

        # Track zH magnitude for sanity (mean over last segment)
        zH_mean = float(torch.norm(zH_list[-1], dim=-1).mean().item()) if len(zH_list) > 0 else 0.0

        def _segment_correctness(logits, prompt_lengths, raw_items):
            pred_ids = logits.argmax(-1)
            outs = []
            for i, item in enumerate(raw_items):
                Lp = prompt_lengths[i]
                text = self.tokenizer.decode(pred_ids[i, Lp:], skip_special_tokens=True)
                ok = item.get("verify", None)
                outs.append(1.0 if callable(ok) and ok(text) else 0.0)
            return torch.tensor(outs, device=logits.device)

        # 4) Deep supervision across segments
        for zH in zH_list:
            injected = self.injector(last_hidden, zH)
            logits = self.logits_from_injected(injected, zH=zH)
            if temperature != 1.0:
                logits = logits / temperature

            loss = ce(logits.view(-1, logits.size(-1)), labels.view(-1))

            # Optional: ACT supervision on zH
            if use_act and self.q_head is not None and isinstance(batch.get("raw", None), list):
                q_logit = self.q_head(zH).squeeze(-1)  # (B,1) → (B,)
                with torch.no_grad():
                    corr = _segment_correctness(logits, prompt_lengths, batch["raw"])
                    y = corr
                q_loss = nn.functional.binary_cross_entropy_with_logits(q_logit, y)
                loss = loss + q_loss + act_penalty
                metrics["q_losses"].append(q_loss.detach().item())

            metrics["segment_losses"].append(loss.detach().item())
            total_loss = total_loss + loss

        metrics["loss"] = float(total_loss.detach().item())
        metrics["zH_mean"] = zH_mean
        return total_loss, metrics

    # -----------------------
    #  Collator
    # -----------------------
    def collate(self, batch, max_length: int = 1024):
        """
        Build (input_ids, attention_mask, labels) for CLM with teacher forcing.
        - The tokenizer encodes prompts and targets separately, then we concat.
        - Prompt tokens in labels are masked with ignore_index.
        - Accepts either 'target' or 'answer' in items. If only 'answer' is present,
          we normalize to the unified target format: '#### <answer>'.
        """
        # Required fields: prompt + (target or answer). Keep raw for verifiers.
        prompts = [b["prompt"] for b in batch]

        targets = []
        for b in batch:
            if "target" in b and isinstance(b["target"], str):
                targets.append(b["target"])
            elif "answer" in b and isinstance(b["answer"], str):
                targets.append(f"#### {b['answer'].strip()}")
            else:
                raise KeyError("Each batch item must include 'target' or 'answer'.")

        # Tokenize separately; enforce a concat-level check below
        enc_p = self.tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=max_length)
        enc_t = self.tokenizer(targets, return_tensors="pt", padding=True, truncation=True, max_length=max_length)

        B = len(batch)
        input_ids, attention_mask, labels, prompt_lengths = [], [], [], []

        for i in range(B):
            p_ids = enc_p["input_ids"][i]
            t_ids = enc_t["input_ids"][i]

            # Drop BOS on target if tokenizer added it (avoid BOS-BOS on concat)
            if len(t_ids) > 0:
                bos_id = getattr(self.tokenizer, "bos_token_id", None)
                if bos_id is not None and t_ids[0].item() == bos_id:
                    t_ids = t_ids[1:]

            # Ensure EOS ends the target to teach "emit number → stop"
            eos_id = getattr(self.tokenizer, "eos_token_id", None)
            if eos_id is not None:
                if (len(t_ids) == 0) or (t_ids[-1].item() != eos_id):
                    t_ids = torch.cat([t_ids, torch.tensor([eos_id], dtype=t_ids.dtype, device=t_ids.device)], dim=0)

            # Concat
            ids = torch.cat([p_ids, t_ids], dim=0)

            # If too long, trim from the LEFT side of the prompt so the target remains intact
            if ids.size(0) > max_length:
                over = ids.size(0) - max_length
                # Keep at least 1 prompt token
                trim_prompt = min(over, max(1, p_ids.size(0)) - 1)
                p_ids = p_ids[trim_prompt:]
                ids = torch.cat([p_ids, t_ids], dim=0)

            am = torch.ones_like(ids)

            lab = ids.clone()
            Lp = p_ids.size(0)
            lab[:Lp] = self.hrm_cfg.vocab_ignore_index  # mask prompt region

            input_ids.append(ids)
            attention_mask.append(am)
            labels.append(lab)
            prompt_lengths.append(int(Lp))

        maxT = max(x.size(0) for x in input_ids)

        def pad_list(x_list, pad_id):
            out = torch.full((B, maxT), pad_id, dtype=torch.long)
            for i, xi in enumerate(x_list):
                out[i, : xi.size(0)] = xi
            return out

        input_ids = pad_list(input_ids, self.tokenizer.pad_token_id or 0)
        attention_mask = pad_list(attention_mask, 0)
        labels = pad_list(labels, self.hrm_cfg.vocab_ignore_index)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "prompt_lengths": prompt_lengths,  # list[int], used for pooling & decode windows
            "raw": batch,                      # keep raw items (contains 'verify', etc.)
        }

    # -----------------------
    #  Checkpoint helpers
    # -----------------------
    def trainable_state_dict(self):
        """
        Return only small trainable parts (HRM + injector + projections + vocab-bias).
        Tokenizer/LLM configs are not saved here—LLM is frozen and reloaded by name/path.
        """
        out = {
            "hrm_cfg": self.hrm_cfg.__dict__,
            "x_proj": self.x_proj.state_dict(),
            "pool_mix": self.pool_mix.state_dict(),
            "h_block": self.h_block.state_dict(),
            "l_block": self.l_block.state_dict(),
            "in_norm": self.in_norm.state_dict(),
            "zH0": self.zH0.detach().cpu(),
            "zL0": self.zL0.detach().cpu(),
            "injector": self.injector.state_dict(),
            "q_head": None if self.q_head is None else self.q_head.state_dict(),
            "tokenizer_name": getattr(self.tokenizer, "name_or_path", None),
            "lm_hidden_size": self.lm.config.hidden_size,
        }
        if self.vocab_bias is not None:
            out["vocab_bias"] = self.vocab_bias.state_dict()
            out["vocab_gate"] = self.vocab_gate.detach().cpu()
        else:
            out["vocab_bias"] = None
            out["vocab_gate"] = None
        return out

    def load_trainable_state_dict(self, payload: dict, strict: bool = True):
        """
        Load small trainable parts saved by trainable_state_dict().
        """
        # If hrm_cfg changed shape (e.g., d_h), you should re-init matching modules.
        self.x_proj.load_state_dict(payload["x_proj"])
        self.pool_mix.load_state_dict(payload["pool_mix"])
        self.h_block.load_state_dict(payload["h_block"])
        self.l_block.load_state_dict(payload["l_block"])
        self.in_norm.load_state_dict(payload["in_norm"])

        with torch.no_grad():
            self.zH0.copy_(payload["zH0"].to(self.zH0.device))
            self.zL0.copy_(payload["zL0"].to(self.zL0.device))

        self.injector.load_state_dict(payload["injector"])
        if self.q_head is not None and payload.get("q_head") is not None:
            self.q_head.load_state_dict(payload["q_head"])

        # Vocab-bias head (if enabled now and present in payload)
        if self.vocab_bias is not None and payload.get("vocab_bias") is not None:
            self.vocab_bias.load_state_dict(payload["vocab_bias"])
        if (self.vocab_gate is not None) and (payload.get("vocab_gate") is not None):
            with torch.no_grad():
                self.vocab_gate.copy_(payload["vocab_gate"].to(self.vocab_gate.device))
