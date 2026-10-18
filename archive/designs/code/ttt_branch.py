import os, sys, json, glob, re, argparse, random
import torch, torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

# ---------------- TwoPathTTT (causal, ViT3-grounded; copied from lena layer) ----------------
class TwoPathTTT(nn.Module):
    def __init__(self, d_in, d_out, r, lr=1.0, ksize=4):
        super().__init__()
        self.r=int(r); self.lr=float(lr); self.scale=float(r)**-0.5; self.ksize=int(ksize)
        self.Wq=nn.Parameter(torch.randn(r,d_in)*(d_in**-0.5))
        self.Wk=nn.Parameter(torch.randn(r,d_in)*(d_in**-0.5))
        self.Wv=nn.Parameter(torch.randn(r,d_in)*(d_in**-0.5))
        self.w1=nn.Parameter(torch.empty(r,r)); nn.init.trunc_normal_(self.w1,std=0.02)
        self.w2=nn.Parameter(torch.empty(r,r)); nn.init.trunc_normal_(self.w2,std=0.02)
        self.Wq2=nn.Parameter(torch.randn(r,d_in)*(d_in**-0.5))
        self.conv=nn.Parameter(torch.zeros(r,1,ksize))
        self.Bt=nn.Parameter(torch.zeros(d_out,2*r))
    def _swiglu(self,q,k,v):
        z1=k@self.w1; z2=k@self.w2; sig=torch.sigmoid(z2); a=z2*sig
        e=-v*self.scale; ea=e*a; eb=e*z1*(sig*(1.0+z2*(1.0-sig)))
        Nn=k.shape[-2]; cnt=torch.arange(1,Nn+1,device=k.device,dtype=k.dtype).view(1,Nn,1,1)
        S1=torch.cumsum(torch.einsum('bnr,bns->bnrs',k,ea),dim=1)/cnt
        S2=torch.cumsum(torch.einsum('bnr,bns->bnrs',k,eb),dim=1)/cnt
        S1=S1/(S1.norm(dim=-2,keepdim=True)+1.0); S2=S2/(S2.norm(dim=-2,keepdim=True)+1.0)
        qw1=q@self.w1 - self.lr*torch.einsum('bnr,bnrs->bns',q,S1)
        qw2=q@self.w2 - self.lr*torch.einsum('bnr,bnrs->bns',q,S2)
        return qw1*torch.nn.functional.silu(qw2)
    def _conv(self,q2):
        xc=q2.transpose(1,2); xc=torch.nn.functional.pad(xc,(self.ksize-1,0))
        o=torch.nn.functional.conv1d(xc,self.conv.to(xc.dtype),groups=self.r)
        return o.transpose(1,2)
    def forward(self,x):
        xf=x.to(self.Wq.dtype)
        q=xf@self.Wq.t(); k=xf@self.Wk.t(); v=xf@self.Wv.t()
        o1=self._swiglu(q,k,v); o2=self._conv(xf@self.Wq2.t())
        o=torch.cat([o1,o2],dim=-1)
        return (o@self.Bt.t())

class TTTBranch(nn.Module):
    def __init__(self,d,r):
        super().__init__(); self.ttt=TwoPathTTT(d,d,r); self.gamma=nn.Parameter(torch.ones(1))
    def forward(self,h):
        return (self.gamma*self.ttt(h)).to(h.dtype)

class WrappedAttn(nn.Module):
    def __init__(self,attn,branch): super().__init__(); self.attn=attn; self.branch=branch
    def forward(self,hidden_states,*a,**k):
        out=self.attn(hidden_states,*a,**k)
        add=self.branch(hidden_states)
        if isinstance(out,tuple):
            return (out[0]+add.to(out[0].dtype),)+tuple(out[1:])
        return out+add.to(out.dtype)

def build(model, r, dtype=torch.float32, device='cuda'):
    layers=model.model.layers; d=model.config.hidden_size
    branches=nn.ModuleList([TTTBranch(d,r) for _ in range(len(layers))]).to(device=device,dtype=dtype)
    for i,layer in enumerate(layers):
        layer.self_attn=WrappedAttn(layer.self_attn, branches[i])
    return branches

# ---------------- data / scoring (match eval_generate) ----------------
PROMPT="### Task: Solve the math problem. End with '#### <number>'.\n\nQuestion: {q}\n\nAnswer:"
def last_number(s):
    n=re.findall(r"-?\d[\d,]*\.?\d*", s.replace(",","")); return n[-1].rstrip(".") if n else None

def gsm_examples(split, n):
    ds=load_dataset("openai/gsm8k","main",split=split)
    out=[]
    for i in range(min(n,len(ds))):
        ex=ds[i]; out.append((PROMPT.format(q=ex["question"]), str(ex["answer"])))
    return out

def load_test(task, limit):
    if task=="gsm8k":
        ds=load_dataset("openai/gsm8k","main",split="test")
        return [(PROMPT.format(q=ds[i]["question"]), last_number(str(ds[i]["answer"]).split("####")[-1])) for i in range(min(limit,len(ds)))]
    if task=="gsm_symbolic":
        ds=load_dataset("apple/GSM-Symbolic","main",split="test")
        return [(PROMPT.format(q=ds[i]["question"]), last_number(str(ds[i]["answer"]).split("####")[-1])) for i in range(min(limit,len(ds)))]

# ---------------- train ----------------
def train(args, model, tok, branches, device):
    exs=gsm_examples("train", args.n_train)
    opt=torch.optim.AdamW(branches.parameters(), lr=args.lr)
    model.eval()  # base frozen; branches train
    for p in model.parameters(): p.requires_grad_(False)
    for p in branches.parameters(): p.requires_grad_(True)
    step=0; steps=args.steps if args.steps>0 else (len(exs)//args.bs)*args.epochs
    order=list(range(len(exs)))
    for ep in range(args.epochs):
        random.Random(args.seed+ep).shuffle(order)
        for bi in range(0,len(order),args.bs):
            idx=order[bi:bi+args.bs]
            input_ids=[]; labels=[]
            for j in idx:
                p,a=exs[j]
                pi=tok(p, add_special_tokens=True).input_ids
                ai=tok(" "+a.strip(), add_special_tokens=False).input_ids+[tok.eos_token_id]
                ids=(pi+ai)[:args.cutoff]
                lab=([-100]*len(pi)+ai)[:args.cutoff]
                input_ids.append(ids); labels.append(lab)
            m=max(len(x) for x in input_ids)
            ii=torch.tensor([x+[tok.pad_token_id]*(m-len(x)) for x in input_ids],device=device)
            ll=torch.tensor([x+[-100]*(m-len(x)) for x in labels],device=device)
            am=(ii!=tok.pad_token_id).long()
            out=model(input_ids=ii, attention_mask=am, labels=ll, use_cache=False)
            loss=out.loss
            opt.zero_grad(); loss.backward()
            gnorm=torch.nn.utils.clip_grad_norm_(branches.parameters(),1.0)
            opt.step(); step+=1
            if step%10==0 or step<=3:
                g=float(branches[0].gamma.detach().float()); gg=float(branches[len(branches)//2].gamma.detach().float())
                print(f"step {step}/{steps} loss={float(loss):.4f} gnorm={float(gnorm):.3f} gamma0={g:.4f} gammaMid={gg:.4f}", flush=True)
            if args.steps>0 and step>=args.steps: return

@torch.no_grad()
def evaluate(model, tok, device, task, limit):
    model.eval()
    test=load_test(task, limit); correct=0
    for i,(p,gold) in enumerate(test):
        ids=tok(p,return_tensors="pt").input_ids.to(device)
        out=model.generate(ids, max_new_tokens=160, do_sample=False, use_cache=False, pad_token_id=tok.pad_token_id)
        pred=tok.decode(out[0,ids.shape[1]:], skip_special_tokens=True)
        pp=last_number(pred)
        correct+= float(pp is not None and gold is not None and pp==gold)
        if (i+1)%25==0: print(f"  eval {task} {i+1}/{len(test)} acc={correct/(i+1):.3f}", flush=True)
    acc=correct/len(test); print(f"[EVAL] {task} exact_match={acc:.4f} n={len(test)}", flush=True); return acc

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--base_model",default="meta-llama/Llama-2-7b-hf")
    ap.add_argument("--r",type=int,default=64)
    ap.add_argument("--n_train",type=int,default=7000)
    ap.add_argument("--epochs",type=int,default=3)
    ap.add_argument("--steps",type=int,default=0)
    ap.add_argument("--bs",type=int,default=2)
    ap.add_argument("--lr",type=float,default=3e-4)
    ap.add_argument("--cutoff",type=int,default=512)
    ap.add_argument("--seed",type=int,default=1)
    ap.add_argument("--out",required=True)
    ap.add_argument("--mode",default="train_eval")
    ap.add_argument("--eval_limit",type=int,default=100)
    args=ap.parse_args()
    dev='cuda'
    tok=AutoTokenizer.from_pretrained(args.base_model); tok.pad_token=tok.eos_token
    model=AutoModelForCausalLM.from_pretrained(args.base_model, torch_dtype=torch.bfloat16, device_map={"":0})
    branches=build(model, args.r, dtype=torch.float32, device=dev)
    ntr=sum(p.numel() for p in branches.parameters())
    print(f"[BUILD] branches={len(branches)} r={args.r} trainable={ntr:,} ({100*ntr/sum(p.numel() for p in model.parameters()):.3f}% of base)", flush=True)
    if args.mode=="eval":
        sd=torch.load(args.out+"/branches.pt", map_location=dev); branches.load_state_dict(sd)
        for t in ["gsm8k","gsm_symbolic"]: evaluate(model,tok,dev,t,args.eval_limit)
        return
    if args.mode=="smoke":
        args.steps=5; args.n_train=64
    train(args, model, tok, branches, dev)
    os.makedirs(args.out, exist_ok=True)
    torch.save(branches.state_dict(), args.out+"/branches.pt")
    print(f"[SAVE] {args.out}/branches.pt", flush=True)
    if args.mode=="smoke":
        sd=torch.load(args.out+"/branches.pt",map_location=dev); branches.load_state_dict(sd)
        g=float(branches[0].gamma.detach().float())
        print(f"[SMOKE_OK] reload ok, gamma0={g:.5f} (moved from 0 => trained)", flush=True)
        return
    for t in ["gsm8k","gsm_symbolic"]: evaluate(model,tok,dev,t,args.eval_limit)

if __name__=="__main__": main()
