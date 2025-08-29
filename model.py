import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Optional, Tuple

from transformers import AutoModelForCausalLM, AutoTokenizer

from hrm_blocks import HBlock, LBlock, RMSNorm
from injector import InjectorGRB, CrossAttentionBridge

# --- Config ---

@dataclass
class HRMConfig:
    d_h: int = 512
    h_layers: int = 1
    l_layers: int = 1
    n_heads: int = 8
    inner_T: int = 3          # low-level steps per segment
    segments: int = 3         # number of segments
    use_cab: bool = False     # injector type
    use_act: bool = False     # halting head enabled
    vocab_ignore_index: int = -100

# --- Controller ---

class HRMController(nn.Module):
    """
    Wraps a frozen HF CausalLM with HRM blocks and an injector.
    Training uses deep supervision across segments and 1-step gradient.
    """
    def __init__(self, model_name: str, hrm_cfg: HRMConfig):
        super().__init__()
        self.hrm_cfg = hrm_cfg

        # Load HF model & tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.lm = AutoModelForCausalLM.from_pretrained(model_name)
        for p in self.lm.parameters():
            p.requires_grad_(False)
        self.lm.eval()

        d_model = self.lm.config.hidden_size

        # HRM blocks
        self.h_block = HBlock(d=hrm_cfg.d_h, n_layers=hrm_cfg.h_layers, n_heads=hrm_cfg.n_heads)
        self.l_block = LBlock(d=hrm_cfg.d_h, n_layers=hrm_cfg.l_layers, n_heads=hrm_cfg.n_heads)
        self.in_norm = RMSNorm(hrm_cfg.d_h)

        # Project pooled token embeddings to HRM width
        self.x_proj = nn.Linear(d_model, hrm_cfg.d_h, bias=False)

        # Initial states (learned)
        self.zH0 = nn.Parameter(torch.zeros(1, 1, hrm_cfg.d_h))
        self.zL0 = nn.Parameter(torch.zeros(1, 1, hrm_cfg.d_h))

        # Injector
        if hrm_cfg.use_cab:
            self.injector = CrossAttentionBridge(d_h=hrm_cfg.d_h, d_model=d_model, n_heads=8)
        else:
            self.injector = InjectorGRB(d_h=hrm_cfg.d_h, d_model=d_model)

        # Optional ACT head (stub)
        self.q_head = nn.Linear(hrm_cfg.d_h, 1) if hrm_cfg.use_act else None

    # --- utils ---

    def pool_tokens(self, hidden_states, mask, prompt_lengths):
        """
        Mean-pool over prompt tokens only (exclude target region).
        hidden_states: (B, T, D)
        prompt_lengths: list[int] length B
        """
        pooled = []
        for i, Lp in enumerate(prompt_lengths):
            hs = hidden_states[i, :Lp, :]  # (Lp, D)
            pooled.append(hs.mean(dim=0))
        return torch.stack(pooled, dim=0).unsqueeze(1)  # (B,1,D)

    def forward_llm_hidden(self, input_ids, attention_mask):
        # No grad through LLM hidden extraction
        with torch.no_grad():
            out = self.lm(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
            last_hidden = out.hidden_states[-1]  # (B,T,D)
        return last_hidden

    def logits_from_injected(self, injected_hidden):
        # Use the LM head to get logits; params are frozen but differentiable ops propagate to injector/HRM.
        lm_head = self.lm.get_output_embeddings()
        logits = lm_head(injected_hidden)  # (B,T,V)
        return logits

    # --- HRM segment loop ---

    def hrm_segments(self, x_tilde, segments: int, inner_T: int):
        """
        Run HRM recurrence with deep supervision + 1-step gradient.
        Returns list of zH states, one per segment (for deep supervision).
        """
        B = x_tilde.size(0)
        zH = self.zH0.expand(B, -1, -1)
        zL = self.zL0.expand(B, -1, -1)
        zH_list = []

        for m in range(segments):
            # roll (T-1) inner steps for L under no-grad; then one grad step for (L,H)
            with torch.no_grad():
                zH_t, zL_t = zH, zL
                for _ in range(inner_T - 1):
                    zL_t = self.l_block(zL_t, zH_t, x_tilde)

            # 1-step gradient: a final low-level update + high-level update
            zL = self.l_block(zL_t, zH_t, x_tilde)
            zH = self.h_block(zH_t, zL)
            zH = self.in_norm(zH)

            zH_list.append(zH)

            # detach between segments (deep supervision)
            zH = zH.detach()
            zL = zL.detach()

        return zH_list

    # --- Training step ---

    def training_step(self, batch, segments=None, inner_T=None, use_act=False):
        """
        batch: dict with 'input_ids', 'attention_mask', 'labels', 'prompt_lengths'
        Returns: total_loss, metrics
        """
        segments = segments or self.hrm_cfg.segments
        inner_T = inner_T or self.hrm_cfg.inner_T

        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]
        prompt_lengths = batch["prompt_lengths"]  # list[int]

        # Extract frozen LLM last hidden states and pooled prompt embedding
        last_hidden = self.forward_llm_hidden(input_ids, attention_mask)  # (B,T,D_model)
        x_pool = self.pool_tokens(last_hidden, attention_mask, prompt_lengths)  # (B,1,D_model)
        x_tilde = self.x_proj(x_pool)  # (B,1,d_h)

        # HRM segments (returns list of zH per segment)
        zH_list = self.hrm_segments(x_tilde, segments=segments, inner_T=inner_T)

        total_loss = 0.0
        ce = nn.CrossEntropyLoss(ignore_index=self.hrm_cfg.vocab_ignore_index)

        metrics = {"segment_losses": []}

        # For each segment, compute deep supervision loss:
        for zH in zH_list:
            # Inject and compute logits
            injected = self.injector(last_hidden, zH)  # (B,T,D_model)
            logits = self.logits_from_injected(injected)  # (B,T,V)

            # Standard CLM teacher forcing: shift and compute CE on labels
            # labels already set with ignore_index for prompt tokens
            loss = ce(logits.view(-1, logits.size(-1)), labels.view(-1))

            # Optional ACT head (stub: wire targets in your trainer if enabling ACT)
            if use_act and self.q_head is not None:
                q_logit = self.q_head(zH).squeeze(-1)  # (B,1)->(B)
                loss = loss + 0.0 * q_logit.mean()

            metrics["segment_losses"].append(loss.detach().item())
            total_loss = total_loss + loss

        metrics["loss"] = total_loss.detach().item()
        return total_loss, metrics

    # --- Tokenization / Collate ---

    def collate(self, batch, max_length=512):
        # batch items: {"prompt": str, "target": str, "verify": callable}
        prompts = [b["prompt"] for b in batch]
        targets = [b["target"] for b in batch]

        # Build input as:  [PROMPT] + [TARGET]
        enc_prompt = self.tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=max_length)
        enc_target = self.tokenizer(targets, return_tensors="pt", padding=True, truncation=True, max_length=max_length)

        B = len(batch)
        input_ids = []
        attention_mask = []
        labels = []
        prompt_lengths = []

        for i in range(B):
            p_ids = enc_prompt["input_ids"][i]
            t_ids = enc_target["input_ids"][i]

            # Combine with BOS handling by tokenizer; simple concat
            ids = torch.cat([p_ids, t_ids], dim=0)
            # attention mask
            am = torch.ones_like(ids)

            # labels: ignore prompt, learn to predict target tokens
            lab = ids.clone()
            prompt_len = p_ids.size(0)
            lab[:prompt_len] = -100  # ignore prompt region for CE

            input_ids.append(ids)
            attention_mask.append(am)
            labels.append(lab)
            prompt_lengths.append(int(prompt_len))

        # pad to same length
        maxT = max(x.size(0) for x in input_ids)
        def pad(x, pad_id):
            out = torch.full((B, maxT), pad_id, dtype=torch.long)
            for i, xi in enumerate(x):
                out[i, :xi.size(0)] = xi
            return out

        input_ids = pad(input_ids, self.tokenizer.pad_token_id or 0)
        attention_mask = pad(attention_mask, 0)
        labels = pad(labels, -100)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "prompt_lengths": prompt_lengths,
            "raw": batch
        }
