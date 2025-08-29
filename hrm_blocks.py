import torch
import torch.nn as nn
from einops import rearrange

# --- Utilities ---

class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-6):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(d))
        self.eps = eps
    def forward(self, x):
        # x: (B, T, D)
        norm_x = x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return self.scale * norm_x

class SwiGLU(nn.Module):
    def __init__(self, d, expansion=4):
        super().__init__()
        hidden = int(d * expansion)
        self.w1 = nn.Linear(d, hidden, bias=False)
        self.w2 = nn.Linear(d, hidden, bias=False)
        self.w3 = nn.Linear(hidden, d, bias=False)
    def forward(self, x):
        return self.w3(torch.nn.functional.silu(self.w1(x)) * self.w2(x))

class SelfAttention(nn.Module):
    def __init__(self, d, n_heads):
        super().__init__()
        assert d % n_heads == 0
        self.d = d
        self.h = n_heads
        self.head_dim = d // n_heads
        self.qkv = nn.Linear(d, 3*d, bias=False)
        self.o = nn.Linear(d, d, bias=False)

    def forward(self, x, mask=None):
        B, T, D = x.shape
        qkv = self.qkv(x)  # (B,T,3D)
        q, k, v = qkv.split(self.d, dim=-1)
        # (B, H, T, Hd)
        q = rearrange(q, 'b t (h d) -> b h t d', h=self.h)
        k = rearrange(k, 'b t (h d) -> b h t d', h=self.h)
        v = rearrange(v, 'b t (h d) -> b h t d', h=self.h)
        # scaled dot-product
        att = torch.matmul(q, k.transpose(-1, -2)) / (self.head_dim ** 0.5)  # (B,H,T,T)
        if mask is not None:
            att = att.masked_fill(mask == 0, float('-inf'))
        att = torch.softmax(att, dim=-1)
        y = torch.matmul(att, v)  # (B,H,T,Hd)
        y = rearrange(y, 'b h t d -> b t (h d)')
        return self.o(y)

class TransformerBlock(nn.Module):
    # Post-Norm Transformer block (RMSNorm) with bias-free linear layers.
    def __init__(self, d, n_heads, mlp_expansion=4):
        super().__init__()
        self.norm1 = RMSNorm(d)
        self.attn = SelfAttention(d, n_heads)
        self.norm2 = RMSNorm(d)
        self.mlp = SwiGLU(d, expansion=mlp_expansion)

    def forward(self, x, attn_mask=None):
        x = x + self.attn(self.norm1(x), mask=attn_mask)
        x = x + self.mlp(self.norm2(x))
        return x

# --- HRM Blocks ---

class HBlock(nn.Module):
    # High-level slow recurrent module (1-2 transformer layers).
    def __init__(self, d=512, n_layers=1, n_heads=8):
        super().__init__()
        self.layers = nn.ModuleList([TransformerBlock(d, n_heads) for _ in range(n_layers)])
        self.norm = RMSNorm(d)

    def forward(self, zH, zL_summary):
        # Combine via simple addition; could be gated/concat.
        x = zH + zL_summary
        for blk in self.layers:
            x = blk(x)
        return self.norm(x)

class LBlock(nn.Module):
    # Low-level fast recurrent module (1-2 transformer layers).
    def __init__(self, d=512, n_layers=1, n_heads=8):
        super().__init__()
        self.layers = nn.ModuleList([TransformerBlock(d, n_heads) for _ in range(n_layers)])
        self.norm = RMSNorm(d)

    def forward(self, zL, zH, x_tilde):
        # Combine three streams by simple addition.
        x = zL + zH + x_tilde
        for blk in self.layers:
            x = blk(x)
        return self.norm(x)
