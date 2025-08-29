import torch
import torch.nn as nn

class InjectorGRB(nn.Module):
    """
    Gated Residual Bias (GRB):
    - Projects zH (B,1,Dh) -> (B,1,D_model) then broadcasts to (B,T,D_model) and adds to token hidden states.
    """
    def __init__(self, d_h=512, d_model=2048):
        super().__init__()
        self.proj = nn.Linear(d_h, d_model, bias=False)
        self.gate = nn.Parameter(torch.tensor(0.1))  # learnable scalar gate

    def forward(self, last_hidden, zH):
        # last_hidden: (B, T, D_model), zH: (B, 1, d_h)
        bias = self.proj(zH)  # (B,1,D_model)
        return last_hidden + torch.sigmoid(self.gate) * bias

class CrossAttentionBridge(nn.Module):
    """
    Cross-Attention Bridge (CAB):
    - Creates a single zH "memory token" and one cross-attention pass on top of the last hidden.
    """
    def __init__(self, d_h=512, d_model=2048, n_heads=8):
        super().__init__()
        self.mem_proj = nn.Linear(d_h, d_model, bias=False)
        self.q = nn.Linear(d_model, d_model, bias=False)
        self.k = nn.Linear(d_model, d_model, bias=False)
        self.v = nn.Linear(d_model, d_model, bias=False)
        self.o = nn.Linear(d_model, d_model, bias=False)
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        assert d_model % n_heads == 0

    def forward(self, last_hidden, zH):
        # last_hidden: (B,T,D), zH: (B,1,d_h)
        B, T, D = last_hidden.shape
        mem = self.mem_proj(zH)  # (B,1,D)

        Q = self.q(last_hidden).view(B, T, self.n_heads, self.head_dim).transpose(1,2)  # (B,H,T,Hd)
        K = self.k(mem).view(B, 1, self.n_heads, self.head_dim).transpose(1,2)         # (B,H,1,Hd)
        V = self.v(mem).view(B, 1, self.n_heads, self.head_dim).transpose(1,2)

        att = (Q @ K.transpose(-1,-2)) / (self.head_dim ** 0.5)  # (B,H,T,1)
        att = torch.softmax(att, dim=-2)  # softmax over T
        out = att @ V  # (B,H,T,Hd)
        out = out.transpose(1,2).contiguous().view(B, T, D)
        return last_hidden + self.o(out)
