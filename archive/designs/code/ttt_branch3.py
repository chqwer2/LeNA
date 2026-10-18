import os, re, argparse, random
import torch, torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

# Ablation TwoPathTTT: inner_lr=0 => pure SwiGLU (TTT off); mid widens the SwiGLU hidden (capacity test).
class TwoPathTTT(nn.Module):
    def __init__(self, d_in, d_out, r, inner_lr=1.0, mid=1, ksize=4):
        super().__init__()
        self.r=int(r); self.inner_lr=float(inner_lr); self.mid=int(mid); self.scale=float(r)**-0.5; self.ksize=int(ksize)
        self.Wq=nn.Parameter(torch.randn(r,d_in)*(d_in**-0.5)); self.Wk=nn.Parameter(torch.randn(r,d_in)*(d_in**-0.5))
        self.Wv=nn.Parameter(torch.randn(r,d_in)*(d_in**-0.5)); self.Wq2=nn.Parameter(torch.randn(r,d_in)*(d_in**-0.5))
        if self.inner_lr>0:   # TTT path needs square r x r fast-weights
            self.w1=nn.Parameter(torch.empty(r,r)); nn.init.trunc_normal_(self.w1,std=0.02)
            self.w2=nn.Parameter(torch.empty(r,r)); nn.init.trunc_normal_(self.w2,std=0.02)
            self.w3=None
        else:                 # pure SwiGLU MLP r -> mid*r -> r
            h=r*self.mid
            self.w1=nn.Parameter(torch.empty(r,h)); nn.init.trunc_normal_(self.w1,std=0.02)
            self.w2=nn.Parameter(torch.empty(r,h)); nn.init.trunc_normal_(self.w2,std=0.02)
            self.w3=nn.Parameter(torch.empty(h,r)); nn.init.trunc_normal_(self.w3,std=0.02)
        self.conv=nn.Parameter(torch.zeros(r,1,ksize)); self.Bt=nn.Parameter(torch.zeros(d_out,2*r))
    def _sg(self,q,k,v):
        if self.inner_lr==0:
            g=(q@self.w1)*torch.nn.functional.silu(q@self.w2); return g@self.w3
        z1=k@self.w1; z2=k@self.w2; sig=torch.sigmoid(z2); a=z2*sig
        e=-v*self.scale; ea=e*a; eb=e*z1*(sig*(1.0+z2*(1.0-sig)))
        Nn=k.shape[-2]; cnt=torch.arange(1,Nn+1,device=k.device,dtype=k.dtype).view(1,Nn,1,1)
        S1=torch.cumsum(torch.einsum('bnr,bns->bnrs',k,ea),dim=1)/cnt; S2=torch.cumsum(torch.einsum('bnr,bns->bnrs',k,eb),dim=1)/cnt
        S1=S1/(S1.norm(dim=-2,keepdim=True)+1.0); S2=S2/(S2.norm(dim=-2,keepdim=True)+1.0)
        qw1=q@self.w1-self.inner_lr*torch.einsum('bnr,bnrs->bns',q,S1); qw2=q@self.w2-self.inner_lr*torch.einsum('bnr,bnrs->bns',q,S2)
        return qw1*torch.nn.functional.silu(qw2)
    def _cv(self,q2):
        xc=q2.transpose(1,2); xc=torch.nn.functional.pad(xc,(self.ksize-1,0))
        return torch.nn.functional.conv1d(xc,self.conv.to(xc.dtype),groups=self.r).transpose(1,2)
    def forward(self,x):
        xf=x.to(self.Wq.dtype); q=xf@self.Wq.t(); k=xf@self.Wk.t(); v=xf@self.Wv.t()
        o=torch.cat([self._sg(q,k,v), self._cv(xf@self.Wq2.t())],dim=-1); return o@self.Bt.t()

class TTTBranch(nn.Module):
    def __init__(self,d,r,inner_lr,mid): super().__init__(); self.ttt=TwoPathTTT(d,d,r,inner_lr,mid); self.gamma=nn.Parameter(torch.ones(1))
    def forward(self,h): return (self.gamma*self.ttt(h)).to(h.dtype)

class WrappedAttn(nn.Module):
    def __init__(self,attn,branch): super().__init__(); self.attn=attn; self.branch=branch
    def forward(self,hidden_states,*a,**k):
        out=self.attn(hidden_states,*a,**k); add=self.branch(hidden_states)
        if isinstance(out,tuple): return (out[0]+add.to(out[0].dtype),)+tuple(out[1:])
        return out+add.to(out.dtype)
class WrappedMLP(nn.Module):
    def __init__(self,mlp,branch): super().__init__(); self.mlp=mlp; self.branch=branch
    def forward(self,hidden_states,*a,**k):
        out=self.mlp(hidden_states,*a,**k); add=self.branch(hidden_states)
        return out+add.to(out.dtype)

def build(model, r, targets, inner_lr, mid, dtype=torch.float32, device='cuda'):
    layers=model.model.layers; d=model.config.hidden_size; allb=nn.ModuleList()
    for layer in layers:
        if 'attn' in targets:
            b=TTTBranch(d,r,inner_lr,mid).to(device=device,dtype=dtype); layer.self_attn=WrappedAttn(layer.self_attn,b); allb.append(b)
        if 'mlp' in targets:
            b=TTTBranch(d,r,inner_lr,mid).to(device=device,dtype=dtype); layer.mlp=WrappedMLP(layer.mlp,b); allb.append(b)
    return allb

PROMPT="### Task: Solve the math problem. End with '#### <number>'.\n\nQuestion: {q}\n\nAnswer:"
def gsm_examples(n):
    ds=load_dataset("openai/gsm8k","main",split="train"); return [(PROMPT.format(q=ds[i]["question"]),str(ds[i]["answer"])) for i in range(min(n,len(ds)))]

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--base_model",default="meta-llama/Llama-2-7b-hf"); ap.add_argument("--r",type=int,default=64)
    ap.add_argument("--targets",default="attn"); ap.add_argument("--inner_lr",type=float,default=1.0); ap.add_argument("--mid",type=int,default=1)
    ap.add_argument("--lr",type=float,default=3e-4); ap.add_argument("--diag_steps",type=int,default=2000)
    ap.add_argument("--n_train",type=int,default=4000); ap.add_argument("--bs",type=int,default=2); ap.add_argument("--cutoff",type=int,default=512); ap.add_argument("--seed",type=int,default=1)
    a=ap.parse_args(); dev='cuda'; targets=[t.strip() for t in a.targets.split(",")]
    tok=AutoTokenizer.from_pretrained(a.base_model); tok.pad_token=tok.eos_token
    model=AutoModelForCausalLM.from_pretrained(a.base_model, torch_dtype=torch.bfloat16, device_map={"":0})
    branches=build(model, a.r, targets, a.inner_lr, a.mid, device=dev)
    ntr=sum(p.numel() for p in branches.parameters())
    print(f"[DIAG] targets={targets} r={a.r} inner_lr={a.inner_lr} mid={a.mid} adamlr={a.lr} trainable={ntr:,}",flush=True)
    exs=gsm_examples(a.n_train)
    opt=torch.optim.AdamW(branches.parameters(), lr=a.lr); model.eval()
    for p in model.parameters(): p.requires_grad_(False)
    for p in branches.parameters(): p.requires_grad_(True)
    order=list(range(len(exs))); random.Random(a.seed).shuffle(order); step=0; win=[]
    while step<a.diag_steps:
        for bi in range(0,len(order),a.bs):
            idx=order[bi:bi+a.bs]; ii=[]; ll=[]
            for j in idx:
                p,ans=exs[j]; pi=tok(p,add_special_tokens=True).input_ids; ai=tok(" "+ans.strip(),add_special_tokens=False).input_ids+[tok.eos_token_id]
                ii.append((pi+ai)[:a.cutoff]); ll.append(([-100]*len(pi)+ai)[:a.cutoff])
            m=max(len(x) for x in ii)
            iis=torch.tensor([x+[tok.pad_token_id]*(m-len(x)) for x in ii],device=dev); lls=torch.tensor([x+[-100]*(m-len(x)) for x in ll],device=dev)
            out=model(input_ids=iis, attention_mask=(iis!=tok.pad_token_id).long(), labels=lls, use_cache=False)
            opt.zero_grad(); out.loss.backward(); torch.nn.utils.clip_grad_norm_(branches.parameters(),1.0); opt.step(); step+=1
            win.append(float(out.loss));
            if len(win)>200: win.pop(0)
            if step%200==0: print(f"  step {step}/{a.diag_steps} smoothed_loss(last{len(win)})={sum(win)/len(win):.4f}",flush=True)
            if step>=a.diag_steps: break
    print(f"[DIAG_DONE] inner_lr={a.inner_lr} mid={a.mid} adamlr={a.lr} FINAL_smoothed_loss={sum(win)/len(win):.4f}",flush=True)
if __name__=="__main__": main()
