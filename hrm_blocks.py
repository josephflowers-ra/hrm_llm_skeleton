
import torch, torch.nn as nn
from einops import rearrange

class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-6):
        super().__init__(); self.scale=nn.Parameter(torch.ones(d)); self.eps=eps
    def forward(self,x): return self.scale * x * torch.rsqrt(x.pow(2).mean(-1,keepdim=True)+self.eps)

class SwiGLU(nn.Module):
    def __init__(self,d,expansion=4):
        super().__init__(); h=int(d*expansion)
        self.w1=nn.Linear(d,h,bias=False); self.w2=nn.Linear(d,h,bias=False); self.w3=nn.Linear(h,d,bias=False)
    def forward(self,x): import torch.nn.functional as F; return self.w3(F.silu(self.w1(x))*self.w2(x))

class SelfAttention(nn.Module):
    def __init__(self,d,h): super().__init__(); assert d%h==0; self.d=d; self.h=h; self.hd=d//h
    def forward(self,x,mask=None):
        qkv=nn.Linear(self.d,3*self.d,bias=False).to(x.device)(x)
        q,k,v=qkv.split(self.d,-1)
        q=rearrange(q,'b t (h d)->b h t d',h=self.h); k=rearrange(k,'b t (h d)->b h t d',h=self.h); v=rearrange(v,'b t (h d)->b h t d',h=self.h)
        att=(q@k.transpose(-1,-2))/(self.hd**0.5); 
        if mask is not None: att=att.masked_fill(mask==0,float('-inf'))
        att=torch.softmax(att,-1); y=att@v; y=rearrange(y,'b h t d->b t (h d)')
        return nn.Linear(self.d,self.d,bias=False).to(x.device)(y)

class TransformerBlock(nn.Module):
    def __init__(self,d,h,mlp_expansion=4):
        super().__init__(); self.n1=RMSNorm(d); self.attn=SelfAttention(d,h); self.n2=RMSNorm(d); self.mlp=SwiGLU(d,mlp_expansion)
    def forward(self,x,mask=None): x=x+self.attn(self.n1(x),mask); x=x+self.mlp(self.n2(x)); return x

class HBlock(nn.Module):
    def __init__(self,d=512,n_layers=1,n_heads=8): super().__init__(); self.layers=nn.ModuleList([TransformerBlock(d,n_heads) for _ in range(n_layers)]); self.norm=RMSNorm(d)
    def forward(self,zH,zLsum):
        x=zH+zLsum
        for blk in self.layers: x=blk(x)
        return self.norm(x)

class LBlock(nn.Module):
    def __init__(self,d=512,n_layers=1,n_heads=8): super().__init__(); self.layers=nn.ModuleList([TransformerBlock(d,n_heads) for _ in range(n_layers)]); self.norm=RMSNorm(d)
    def forward(self,zL,zH,x_tilde):
        x=zL+zH+x_tilde
        for blk in self.layers: x=blk(x)
        return self.norm(x)
