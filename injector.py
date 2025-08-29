# injector.py
import torch
import torch.nn as nn


class InjectorGRB(nn.Module):
    """
    Gated Residual Bias (GRB)
    Adds a projected bias from zH to every token's hidden state, scaled by a learned gate.

    Inputs:
      last_hidden : (B, T, D)   - LLM final hidden states
      zH          : (B, 1, d_h) - HRM high-level state

    Returns:
      (B, T, D)
    """
    def __init__(self, d_h: int = 512, d_model: int = 2048):
        super().__init__()
        self.proj = nn.Linear(d_h, d_model, bias=False)
        # Learned scalar gate; sigmoid(gate) ∈ (0,1)
        self.gate = nn.Parameter(torch.tensor(0.1))

    def forward(self, last_hidden: torch.Tensor, zH: torch.Tensor) -> torch.Tensor:
        # bias: (B, 1, D) broadcasts across time dimension T
        bias = self.proj(zH)
        return last_hidden + torch.sigmoid(self.gate) * bias


class CrossAttentionBridge(nn.Module):
    """
    Single-hop cross-attention from token states (queries) to a 1-token memory
    derived from zH (keys/values). Output is projected and added back (residual),
    scaled by a learned sigmoid gate (tunable strength).

    Inputs:
      last_hidden : (B, T, D)   - LLM final hidden states
      zH          : (B, 1, d_h) - HRM high-level state

    Returns:
      (B, T, D)
    """
    def __init__(self, d_h: int = 512, d_model: int = 2048, n_heads: int = 8):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.h = n_heads
        self.hd = d_model // n_heads

        # Project zH → model dim to create a single memory token
        self.mem = nn.Linear(d_h, d_model, bias=False)

        # Standard QKV projections and output projection
        self.q = nn.Linear(d_model, d_model, bias=False)
        self.k = nn.Linear(d_model, d_model, bias=False)
        self.v = nn.Linear(d_model, d_model, bias=False)
        self.o = nn.Linear(d_model, d_model, bias=False)

        # Learned scalar gate; sigmoid(gate) ∈ (0,1)
        self.gate = nn.Parameter(torch.tensor(0.1))

    def forward(self, last_hidden: torch.Tensor, zH: torch.Tensor) -> torch.Tensor:
        B, T, D = last_hidden.shape

        # Memory token from zH: (B, 1, D)
        mem = self.mem(zH)

        # Project queries/keys/values
        # Q: (B, h, T, hd);  K,V: (B, h, 1, hd)
        Q = self.q(last_hidden).view(B, T, self.h, self.hd).transpose(1, 2)
        K = self.k(mem).view(B, 1, self.h, self.hd).transpose(1, 2)
        V = self.v(mem).view(B, 1, self.h, self.hd).transpose(1, 2)

        # Scaled dot-product attention
        # att_logits: (B, h, T, 1)
        att_logits = (Q @ K.transpose(-1, -2)) / (self.hd ** 0.5)

        # Softmax over KEYS dimension (last dim)
        att = torch.softmax(att_logits, dim=-1)  # (B, h, T, 1)

        # Weighted sum over V, then merge heads → (B, T, D)
        out = att @ V                               # (B, h, T, hd)
        out = out.transpose(1, 2).contiguous().view(B, T, D)
        out = self.o(out)                           # (B, T, D)

        # Residual with learned sigmoid gate
        return last_hidden + torch.sigmoid(self.gate) * out
