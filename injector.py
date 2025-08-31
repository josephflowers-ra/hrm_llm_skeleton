#!/usr/bin/env python3
import torch
import torch.nn as nn


class InjectorGRB(nn.Module):
    """
    Gated Residual Bias (GRB)
    Adds a projected bias from zH to each token's hidden state, scaled by a learned gate.

    Inputs:
      last_hidden : (B, T, D)   - LLM final hidden states
      zH          : (B, 1, d_h) - HRM high-level state
    Returns:
      (B, T, D)
    """
    def __init__(self, d_h: int = 512, d_model: int = 2048, gate_init: float = -2.0):
        super().__init__()
        self.proj = nn.Linear(d_h, d_model, bias=False)
        # Learned scalar gate; sigmoid(gate) ∈ (0,1).
        self.gate = nn.Parameter(torch.tensor(gate_init))  # e.g., -2 ≈ 0.12

    def forward(self, last_hidden: torch.Tensor, zH: torch.Tensor) -> torch.Tensor:
        # zH: (B,1,d_h) -> bias: (B,1,D) -> broadcast across T
        bias = self.proj(zH)
        return last_hidden + torch.sigmoid(self.gate) * bias


class CrossAttentionBridge(nn.Module):
    """
    Cross-Attention Bridge (CAB) with multi-token memory.

    Single-hop cross-attention from token states (queries) to a small bank of
    memory tokens derived from zH (keys/values). Output is projected and added
    back (residual), scaled by a learned sigmoid gate.

    Inputs:
      last_hidden : (B, T, D)   - LLM final hidden states
      zH          : (B, 1, d_h) - HRM high-level state
    Returns:
      (B, T, D)
    """
    def __init__(
        self,
        d_h: int = 512,
        d_model: int = 2048,
        n_heads: int = 8,
        mem_tokens: int = 4,      # NEW: number of zH-derived memory tokens
        gate_init: float = -1.5,  # slightly stronger than -2.0 (~0.18 vs 0.12)
    ):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.h = n_heads
        self.hd = d_model // n_heads
        self.m = int(mem_tokens)

        # Project zH → m memory tokens in model dim
        # zH: (B,1,d_h) -> mem: (B,m,D)
        self.mem = nn.Linear(d_h, d_model * self.m, bias=False)

        # Standard QKV projections and output projection
        self.q = nn.Linear(d_model, d_model, bias=False)
        self.k = nn.Linear(d_model, d_model, bias=False)
        self.v = nn.Linear(d_model, d_model, bias=False)
        self.o = nn.Linear(d_model, d_model, bias=False)

        # Learned scalar gate to control residual strength
        self.gate = nn.Parameter(torch.tensor(gate_init))

    def forward(self, last_hidden: torch.Tensor, zH: torch.Tensor) -> torch.Tensor:
        """
        Shapes:
          last_hidden: (B,T,D)
          zH:          (B,1,d_h)
        """
        B, T, D = last_hidden.shape

        # ---- Build memory bank from zH ----
        mem = self.mem(zH)                        # (B,1,D*m)
        mem = mem.view(B, self.m, D)              # (B,m,D)

        # ---- Project to multi-head Q, K, V ----
        # Q: (B,h,T,hd); K,V: (B,h,m,hd)
        Q = self.q(last_hidden).view(B, T, self.h, self.hd).transpose(1, 2)
        K = self.k(mem).view(B, self.m, self.h, self.hd).transpose(1, 2)
        V = self.v(mem).view(B, self.m, self.h, self.hd).transpose(1, 2)

        # ---- Attention over memory tokens ----
        # logits: (B,h,T,m); softmax over m
        att_logits = (Q @ K.transpose(-1, -2)) / (self.hd ** 0.5)
        att = torch.softmax(att_logits, dim=-1)

        # ---- Aggregate and project back ----
        out = att @ V                              # (B,h,T,hd)
        out = out.transpose(1, 2).contiguous().view(B, T, D)
        out = self.o(out)                          # (B,T,D)

        # ---- Gated residual ----
        return last_hidden + torch.sigmoid(self.gate) * out
