import os, re, argparse, random
import torch, torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

# Strengthened TTT branch: (1) LEARNABLE per-layer inner_lr, (2) per-channel gate, (3) enriched nonlinear core.
class TwoPathTTT(nn.Module):
    def __init__(self, d_in, d_out, r, ksize=4, core_mlp=True):
        super().__init__()
        self.r=int(r); self.scale=float(r)**-0.5; self.ksize=int(ksize)
        self.Wq=nn.Parameter(torch.randn(r,d_in)*(d_in**-0.5)); self.Wk=nn.Parameter(torch.randn(r,d_in)*(d_in**-0.5))
        self.Wv=nn.Parameter(torch.randn(r,d_in)*(d_in**-0.5)); self.Wq2=nn.Parameter(torch.randn(r,d_in)*(d_in**-0.5))
        self.w1=nn.Parameter(torch.empty(r,r)); nn.init.trunc_normal_(self.w1,std=0.02)
        self.w2=nn.Parameter(torch.empty(r,r)); nn.init.trunc_normal_(self.w2,std=0.02)
        self.ilr=nn.Parameter(torch.tensor(1.0))                       # (1) LEARNABLE inner-loop step
        self.core_mlp=core_mlp
        if core_mlp:                                                   # (3) enrich the cramped nonlinear core: r->2r->r
            self.ce=nn.Parameter(torch.empty(r,2*r)); nn.init.trunc_normal_(self.ce,std=0.02)
            self.cc=nn.Parameter(torch.zeros(2*r,r))                   # zero-init => starts as identity-ish no-op add
        self.conv=nn.Parameter(torch.zeros(r,1,ksize)); self.Bt=nn.Parameter(torch.zeros(d_out,2*r))
    def _sg(self,q,k,v):
        z1=k@self.w1; z2=k@self.w2; sig=torch.sigmoid(z2); a=z2*sig
        e=-v*self.scale; ea=e*a; eb=e*z1*(sig*(1.0+z2*(1.0-sig)))
        Nn=k.shape[-2]; cnt=torch.arange(1,Nn+1,device=k.device,dtype=k.dtype).view(1,Nn,1,1)
        S1=torch.cumsum(torch.einsum('bnr,bns->bnrs',k,ea),dim=1)/cnt; S2=torch.cumsum(torch.einsum('bnr,bns->bnrs',k,eb),dim=1)/cnt
        S1=S1/(S1.norm(dim=-2,keepdim=True)+1.0); S2=S2/(S2.norm(dim=-2,keepdim=True)+1.0)
        qw1=q@self.w1-self.ilr*torch.einsum('bnr,bnrs->bns',q,S1); qw2=q@self.w2-self.ilr*torch.einsum('bnr,bnrs->bns',q,S2)
        core=qw1*torch.nn.functional.silu(qw2)
        if self.core_mlp:
            core=core + (torch.nn.functional.silu(core@self.ce))@self.cc   # residual enriched core (zero-init => safe)
        return core
    def _cv(self,q2):
        xc=q2.transpose(1,2); xc=torch.nn.functional.pad(xc,(self.ksize-1,0))
        return torch.nn.functional.conv1d(xc,self.conv.to(xc.dtype),groups=self.r).transpose(1,2)
    def forward(self,x):
        xf=x.to(self.Wq.dtype); q=xf@self.Wq.t(); k=xf@self.Wk.t(); v=xf@self.Wv.t()
        o=torch.cat([self._sg(q,k,v), self._cv(xf@self.Wq2.t())],dim=-1); return o@self.Bt.t()

class TTTBranch(nn.Module):
    def __init__(self,d,r): super().__init__(); self.ttt=TwoPathTTT(d,d,r); self.gamma=nn.Parameter(torch.ones(d))  # (2) per-channel gate
    def forward(self,h): return (self.gamma*self.ttt(h)).to(h.dtype)

class WrappedAttn(nn.Module):
    def __init__(self,attn,branch): super().__init__(); self.attn=attn; self.branch=branch
    def forward(self,hidden_states,*a,**k):
        out=self.attn(hidden_states,*a,**k); add=self.branch(hidden_states)
        if isinstance(out,tuple): return (out[0]+add.to(out[0].dtype),)+tuple(out[1:])
        return out+add.to(out.dtype)
class WrappedMLP(nn.Module):
    def __init__(self,mlp,branch): super().__init__(); self.mlp=mlp; self.branch=branch
    def forward(self,hs,*a,**k):
        out=self.mlp(hs,*a,**k); return out+self.branch(hs).to(out.dtype)

def build(model, r, targets, dtype=torch.float32, device='cuda'):
    layers=model.model.layers; d=model.config.hidden_size; allb=nn.ModuleList()
    for layer in layers:
        if 'attn' in targets:
            b=TTTBranch(d,r).to(device=device,dtype=dtype); layer.self_attn=WrappedAttn(layer.self_attn,b); allb.append(b)
        if 'mlp' in targets:
            b=TTTBranch(d,r).to(device=device,dtype=dtype); layer.mlp=WrappedMLP(layer.mlp,b); allb.append(b)
    return allb

def ln(s):
    n=re.findall(r"-?\d[\d,]*\.?\d*", str(s).replace(",","")); return n[-1].rstrip(".") if n else None
def norm(x): return None if x is None else str(x).replace(",","").strip().rstrip(".")
def extract(g):
    m=re.search(r"####\s*\$?\s*(-?[\d][\d,]*\.?\d*)", g); return norm(m.group(1)) if m else norm(ln(g))
PROMPT="### Task: Solve the math problem. End with '#### <number>'.\n\nQuestion: {q}\n\nAnswer:"
def gsm_examples(n):
    ds=load_dataset("openai/gsm8k","main",split="train"); return [(PROMPT.format(q=ds[i]["question"]),str(ds[i]["answer"])) for i in range(min(n,len(ds)))]
def load_test(task,limit):
    if task=="gsm8k":
        ds=load_dataset("openai/gsm8k","main",split="test"); return [(PROMPT.format(q=ds[i]["question"]),norm(ln(str(ds[i]["answer"]).split("####")[-1]))) for i in range(min(limit,len(ds)))]
    if task=="gsm_symbolic":
        ds=load_dataset("apple/GSM-Symbolic","main",split="test"); seen=set(); o=[]
        for i in range(len(ds)):
            k=ds[i].get("original_id",ds[i].get("original_question"))
            if k in seen: continue
            seen.add(k); o.append((PROMPT.format(q=ds[i]["question"]),norm(ln(str(ds[i]["answer"]).split("####")[-1]))))
            if len(o)>=limit: break
        return o

def train(a,model,tok,branches,dev):
    exs=gsm_examples(a.n_train); opt=torch.optim.AdamW(branches.parameters(),lr=a.lr); model.eval()
    for p in model.parameters(): p.requires_grad_(False)
    for p in branches.parameters(): p.requires_grad_(True)
    step=0; steps=(len(exs)//a.bs)*a.epochs; order=list(range(len(exs)))
    for ep in range(a.epochs):
        random.Random(a.seed+ep).shuffle(order)
        for bi in range(0,len(order),a.bs):
            idx=order[bi:bi+a.bs]; ii=[]; ll=[]
            for j in idx:
                p,ans=exs[j]; pi=tok(p,add_special_tokens=True).input_ids; ai=tok(" "+ans.strip(),add_special_tokens=False).input_ids+[tok.eos_token_id]
                ii.append((pi+ai)[:a.cutoff]); ll.append(([-100]*len(pi)+ai)[:a.cutoff])
            m=max(len(x) for x in ii)
            iis=torch.tensor([x+[tok.pad_token_id]*(m-len(x)) for x in ii],device=dev); lls=torch.tensor([x+[-100]*(m-len(x)) for x in ll],device=dev)
            out=model(input_ids=iis,attention_mask=(iis!=tok.pad_token_id).long(),labels=lls,use_cache=False)
            opt.zero_grad(); out.loss.backward(); torch.nn.utils.clip_grad_norm_(branches.parameters(),1.0); opt.step(); step+=1
            if step%20==0 or step<=3:
                il=float(branches[0].ttt.ilr); print(f"step {step}/{steps} loss={float(out.loss):.4f} ilr0={il:.3f}",flush=True)

@torch.no_grad()
def evaluate(model,tok,dev,task,limit,maxtok=128):
    model.eval(); test=load_test(task,limit); c=0
    for p,gold in test:
        ids=tok(p,return_tensors="pt").input_ids.to(dev)
        out=model.generate(ids,max_new_tokens=maxtok,do_sample=False,use_cache=False,pad_token_id=tok.pad_token_id)
        if extract(tok.decode(out[0,ids.shape[1]:],skip_special_tokens=True))==gold: c+=1
    print(f"[EVAL] {task} exact_match={c/len(test):.4f} n={len(test)}",flush=True)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--base_model",default="meta-llama/Llama-2-7b-hf"); ap.add_argument("--r",type=int,default=64); ap.add_argument("--targets",default="attn")
    ap.add_argument("--n_train",type=int,default=7000); ap.add_argument("--epochs",type=int,default=3); ap.add_argument("--bs",type=int,default=2)
    ap.add_argument("--lr",type=float,default=3e-4); ap.add_argument("--cutoff",type=int,default=512); ap.add_argument("--seed",type=int,default=1)
    ap.add_argument("--out",required=True); ap.add_argument("--mode",default="train_eval"); ap.add_argument("--eval_limit",type=int,default=200); ap.add_argument("--eval_tasks",default="gsm8k")
    a=ap.parse_args(); dev='cuda'; targets=[t.strip() for t in a.targets.split(",")]
    tok=AutoTokenizer.from_pretrained(a.base_model); tok.pad_token=tok.eos_token
    model=AutoModelForCausalLM.from_pretrained(a.base_model,torch_dtype=torch.bfloat16,device_map={"":0})
    branches=build(model,a.r,targets,device=dev)
    ntr=sum(p.numel() for p in branches.parameters())
    print(f"[BUILD-v4] targets={targets} r={a.r} trainable={ntr:,} (learnable ilr + per-channel gate + enriched core)",flush=True)
    if a.mode=="smoke":
        exs=gsm_examples(64); opt=torch.optim.AdamW(branches.parameters(),lr=a.lr)
        for p in model.parameters(): p.requires_grad_(False)
        for st in range(5):
            p,ans=exs[st]; pi=tok(p).input_ids; ai=tok(" "+ans.strip(),add_special_tokens=False).input_ids+[tok.eos_token_id]
            ids=torch.tensor([pi+ai],device=dev); lab=torch.tensor([[-100]*len(pi)+ai],device=dev)
            o=model(input_ids=ids,labels=lab,use_cache=False); opt.zero_grad(); o.loss.backward(); opt.step()
            print(f"smoke {st} loss={float(o.loss):.3f} ilr0={float(branches[0].ttt.ilr):.4f}",flush=True)
        print("[SMOKE_OK]",flush=True); return
    train(a,model,tok,branches,dev)
    os.makedirs(a.out,exist_ok=True); torch.save(branches.state_dict(),a.out+"/branches.pt"); print(f"[SAVE] {a.out}",flush=True)
    for t in a.eval_tasks.split(","): evaluate(model,tok,dev,t.strip(),a.eval_limit)
if __name__=="__main__": main()
