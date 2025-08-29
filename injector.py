
import torch, torch.nn as nn

class InjectorGRB(nn.Module):
    def __init__(self,d_h=512,d_model=2048): super().__init__(); self.proj=nn.Linear(d_h,d_model,bias=False); self.gate=nn.Parameter(torch.tensor(0.1))
    def forward(self,last_hidden,zH):
        bias=self.proj(zH); return last_hidden + torch.sigmoid(self.gate)*bias

class CrossAttentionBridge(nn.Module):
    def __init__(self,d_h=512,d_model=2048,n_heads=8):
        super().__init__(); self.mem=nn.Linear(d_h,d_model,bias=False)
        self.q=nn.Linear(d_model,d_model,bias=False); self.k=nn.Linear(d_model,d_model,bias=False); self.v=nn.Linear(d_model,d_model,bias=False); self.o=nn.Linear(d_model,d_model,bias=False)
        self.h=n_heads; self.hd=d_model//n_heads
    def forward(self,last_hidden,zH):
        B,T,D=last_hidden.shape; mem=self.mem(zH)
        Q=self.q(last_hidden).view(B,T,self.h,self.hd).transpose(1,2)
        K=self.k(mem).view(B,1,self.h,self.hd).transpose(1,2)
        V=self.v(mem).view(B,1,self.h,self.hd).transpose(1,2)
        att=(Q@K.transpose(-1,-2))/(self.hd**0.5); att=torch.softmax(att,dim=-2); out=att@V; out=out.transpose(1,2).contiguous().view(B,T,D)
        return last_hidden + self.o(out)
