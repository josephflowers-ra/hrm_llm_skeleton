#!/usr/bin/env python3
import torch
import torch.nn as nn
from dataclasses import dataclass
from transformers import AutoModelForCausalLM, AutoTokenizer

from hrm_blocks import HBlock, LBlock, RMSNorm
from injector import InjectorGRB, CrossAttentionBridge


@dataclass
class HRMConfig:
    # HRM/controller config
    d_h: int = 512
    h_layers: int = 1
    l_layers: int = 1
    n_heads: int = 8
    inner_T: int = 3
    segments: int = 3
    use_cab: bool = False     # False → GRB, True → CAB
    use_act: bool = False
    vocab_ignore_index: int = -100


class HRMController(nn.Module):
    """
    Hybrid controller:
      - Frozen HF CausalLM provides language 'speaker'
      - HRM (H/L blocks) performs latent reasoning on pooled prompt features
      - Injector (GRB or CAB) conditions the LLM hidden states with zH
    Only HRM + injector (and small projections) are trainable.
    """
    def __init__(self, model_name: str, hrm_cfg: HRMConfig):
        super().__init__()
        self.hrm_cfg = hrm_cfg

        # --- Load tokenizer / LM and freeze LM ---
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.lm = AutoModelForCausalLM.from_pretrained(model_name)
        for p in self.lm.parameters():
            p.requires_grad_(False)
        self.lm.eval()

        d_model = self.lm.config.hidden_size

        # --- HRM blocks (trainable) ---
        self.h_block = HBlock(d=hrm_cfg.d_h, n_layers=hrm_cfg.h_layers, n_heads=hrm_cfg.n_heads)
        self.l_block = LBlock(d=hrm_cfg.d_h, n_layers=hrm_cfg.l_layers, n_heads=hrm_cfg.n_heads)
        self.in_norm = RMSNorm(hrm_cfg.d_h)

        # project pooled LLM features → HRM width
        self.x_proj = nn.Linear(d_model, hrm_cfg.d_h, bias=False)

        # learnable initial states (per batch expanded)
        self.zH0 = nn.Parameter(torch.zeros(1, 1, hrm_cfg.d_h))
        self.zL0 = nn.Parameter(torch.zeros(1, 1, hrm_cfg.d_h))

        # --- Injector: GRB or CAB (trainable) ---
        if hrm_cfg.use_cab:
            self.injector = CrossAttentionBridge(d_h=hrm_cfg.d_h, d_model=d_model, n_heads=hrm_cfg.n_heads)
        else:
            self.injector = InjectorGRB(d_h=hrm_cfg.d_h, d_model=d_model)

        # Optional ACT (halt) head on zH
        self.q_head = nn.Linear(hrm_cfg.d_h, 1) if hrm_cfg.use_act else None

    # -----------------------
    #  LLM helper methods
    # -----------------------
    def forward_llm_hidden(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Forward the frozen LLM and return last hidden states (B, T, D)."""
        with torch.no_grad():
            out = self.lm(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
            return out.hidden_states[-1]

    def logits_from_injected(self, injected_hidden: torch.Tensor) -> torch.Tensor:
        """Apply LM head to the conditioned hidden states → logits (B, T, V)."""
        return self.lm.get_output_embeddings()(injected_hidden)

    # -----------------------
    #  HRM dataflow
    # -----------------------
    def pool_tokens(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor, prompt_lengths) -> torch.Tensor:
        """
        Mean-pool ONLY the prompt region per sample → (B, 1, D_model).
        prompt_lengths[i] gives Lp for sample i.
        """
        pooled = []
        for i, Lp in enumerate(prompt_lengths):
            pooled.append(hidden_states[i, :Lp, :].mean(0))
        return torch.stack(pooled, 0).unsqueeze(1)  # (B,1,D_model)

    def hrm_segments(self, x_tilde: torch.Tensor, segments: int, inner_T: int):
        """
        Run HRM for `segments` cycles.
        - First inner_T-1 L-steps under no_grad (let L converge)
        - Final L-step + one H-step with grad
        - Deep supervision: detach(zH,zL) between segments
        Returns list of zH per segment.
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

            # final grad-carrying updates
            zL = self.l_block(zL_t, zH_t, x_tilde)
            zH = self.h_block(zH_t, zL)
            zH = self.in_norm(zH)

            zH_list.append(zH)
            # detach between segments (O(1) memory)
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
        segments = segments or self.hrm_cfg.segments
        inner_T = inner_T or self.hrm_cfg.inner_T

        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]
        prompt_lengths = batch["prompt_lengths"]

        # 1) Frozen LLM forward to get hidden states
        last_hidden = self.forward_llm_hidden(input_ids, attention_mask)

        # 2) Pool prompt → project to HRM width
        x_pool = self.pool_tokens(last_hidden, attention_mask, prompt_lengths)
        x_tilde = self.x_proj(x_pool)

        # 3) HRM loop
        zH_list = self.hrm_segments(x_tilde, segments=segments, inner_T=inner_T)

        ce = nn.CrossEntropyLoss(ignore_index=self.hrm_cfg.vocab_ignore_index)
        metrics = {"segment_losses": [], "q_losses": []}
        total_loss = 0.0

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
            logits = self.logits_from_injected(injected)
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
        return total_loss, metrics

    # -----------------------
    #  Collator
    # -----------------------
    def collate(self, batch, max_length: int = 512):
        """
        Builds input_ids, attention_mask, labels for CLM with teacher forcing.
        Prompt tokens are masked in labels with -100.
        """
        prompts = [b["prompt"] for b in batch]
        targets = [b["target"] for b in batch]

        enc_p = self.tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=max_length)
        enc_t = self.tokenizer(targets, return_tensors="pt", padding=True, truncation=True, max_length=max_length)

        B = len(batch)
        input_ids, attention_mask, labels, prompt_lengths = [], [], [], []

        for i in range(B):
            p_ids = enc_p["input_ids"][i]
            t_ids = enc_t["input_ids"][i]
            ids = torch.cat([p_ids, t_ids], 0)
            am = torch.ones_like(ids)
            lab = ids.clone()
            Lp = p_ids.size(0)
            lab[:Lp] = self.hrm_cfg.vocab_ignore_index
            input_ids.append(ids)
            attention_mask.append(am)
            labels.append(lab)
            prompt_lengths.append(int(Lp))

        maxT = max(x.size(0) for x in input_ids)

        def pad(x, pad_id):
            out = torch.full((B, maxT), pad_id, dtype=torch.long)
            for i, xi in enumerate(x):
                out[i, : xi.size(0)] = xi
            return out

        input_ids = pad(input_ids, self.tokenizer.pad_token_id or 0)
        attention_mask = pad(attention_mask, 0)
        labels = pad(labels, self.hrm_cfg.vocab_ignore_index)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "prompt_lengths": prompt_lengths,
            "raw": batch,  # keep for verifier
        }

    # -----------------------
    #  Checkpoint helpers
    # -----------------------
    def trainable_state_dict(self):
        """Return only small trainable parts (HRM + injector + projections)."""
        return {
            "hrm_cfg": self.hrm_cfg.__dict__,
            "x_proj": self.x_proj.state_dict(),
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

    def load_trainable_state_dict(self, payload: dict, strict: bool = True):
        """Load small trainable parts."""
        self.x_proj.load_state_dict(payload["x_proj"])
        self.h_block.load_state_dict(payload["h_block"])
        self.l_block.load_state_dict(payload["l_block"])
        self.in_norm.load_state_dict(payload["in_norm"])
        with torch.no_grad():
            self.zH0.copy_(payload["zH0"].to(self.zH0.device))
            self.zL0.copy_(payload["zL0"].to(self.zL0.device))
        self.injector.load_state_dict(payload["injector"])
        if self.q_head is not None and payload.get("q_head") is not None:
            self.q_head.load_state_dict(payload["q_head"])
