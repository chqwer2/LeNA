import os, re, argparse, random
import torch, torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

class TwoPathTTT(nn.Module):
    def __init__(self, d_in, d_out, r, lr=1.0, ksize=4):
        super().__init__()
        self.r=int(r); self.lr=float(lr); self.scale=float(r)**-0.5; self.ksize=int(ksize)
        self.Wq=nn.Parameter(torch.randn(r,d_in)*(d_in**-0.5)); self.Wk=nn.Parameter(torch.randn(r,d_in)*(d_in**-0.5))
        self.Wv=nn.Parameter(torch.randn(r,d_in)*(d_in**-0.5)); self.Wq2=nn.Parameter(torch.randn(r,d_in)*(d_in**-0.5))
        self.w1=nn.Parameter(torch.empty(r,r)); nn.init.trunc_normal_(self.w1,std=0.02)
        self.w2=nn.Parameter(torch.empty(r,r)); nn.init.trunc_normal_(self.w2,std=0.02)
        self.conv=nn.Parameter(torch.zeros(r,1,ksize)); self.Bt=nn.Parameter(torch.zeros(d_out,2*r))
    def _sg(self,q,k,v):
        z1=k@self.w1; z2=k@self.w2; sig=torch.sigmoid(z2); a=z2*sig
        e=-v*self.scale; ea=e*a; eb=e*z1*(sig*(1.0+z2*(1.0-sig)))
        Nn=k.shape[-2]; cnt=torch.arange(1,Nn+1,device=k.device,dtype=k.dtype).view(1,Nn,1,1)
        S1=torch.cumsum(torch.einsum('bnr,bns->bnrs',k,ea),dim=1)/cnt; S2=torch.cumsum(torch.einsum('bnr,bns->bnrs',k,eb),dim=1)/cnt
        S1=S1/(S1.norm(dim=-2,keepdim=True)+1.0); S2=S2/(S2.norm(dim=-2,keepdim=True)+1.0)
        qw1=q@self.w1-self.lr*torch.einsum('bnr,bnrs->bns',q,S1); qw2=q@self.w2-self.lr*torch.einsum('bnr,bnrs->bns',q,S2)
        return qw1*torch.nn.functional.silu(qw2)
    def _cv(self,q2):
        xc=q2.transpose(1,2); xc=torch.nn.functional.pad(xc,(self.ksize-1,0))
        return torch.nn.functional.conv1d(xc,self.conv.to(xc.dtype),groups=self.r).transpose(1,2)
    def forward(self,x):
        xf=x.to(self.Wq.dtype); q=xf@self.Wq.t(); k=xf@self.Wk.t(); v=xf@self.Wv.t()
        o=torch.cat([self._sg(q,k,v), self._cv(xf@self.Wq2.t())],dim=-1); return o@self.Bt.t()

class TTTBranch(nn.Module):
    def __init__(self,d,r): super().__init__(); self.ttt=TwoPathTTT(d,d,r); self.gamma=nn.Parameter(torch.ones(1))
    def forward(self,h): return (self.gamma*self.ttt(h)).to(h.dtype)

class WrappedAttn(nn.Module):
    def __init__(self,attn,branch): super().__init__(); self.attn=attn; self.branch=branch
    def forward(self,hidden_states,*a,**k):
        out=self.attn(hidden_states,*a,**k); add=self.branch(hidden_states)
        if isinstance(out,tuple): return (out[0]+add.to(out[0].dtype),)+tuple(out[1:])
        return out+add.to(out.dtype)

def build(model,r,dtype=torch.float32,device='cuda'):
    layers=[m for m in model.modules() if type(m).__name__=="LlamaDecoderLayer"]
    d=model.config.hidden_size
    branches=nn.ModuleList([TTTBranch(d,r) for _ in range(len(layers))]).to(device=device,dtype=dtype)
    for i,layer in enumerate(layers): layer.self_attn=WrappedAttn(layer.self_attn,branches[i])
    return branches

PROMPT="### Task: Solve the math problem. End with '#### <number>'.\n\nQuestion: {q}\n\nAnswer:"
def ln(s):
    n=re.findall(r"-?\d[\d,]*\.?\d*",str(s).replace(",","")); return n[-1].rstrip(".") if n else None
def extract(g):
    m=re.search(r"####\s*\$?\s*(-?[\d][\d,]*\.?\d*)",g)
    return m.group(1).replace(",","").rstrip(".") if m else ln(g)
def gsm_examples(split,n):
    ds=load_dataset("openai/gsm8k","main",split=split); return [(PROMPT.format(q=ds[i]["question"]),str(ds[i]["answer"])) for i in range(min(n,len(ds)))]
def load_test(task,n):
    if task=="gsm8k":
        ds=load_dataset("openai/gsm8k","main",split="test"); return [(PROMPT.format(q=ds[i]["question"]),ln(str(ds[i]["answer"]).split("####")[-1])) for i in range(min(n,len(ds)))]
    if task=="gsm_symbolic":
        ds=load_dataset("apple/GSM-Symbolic","main",split="test"); seen=set(); o=[]
        for i in range(len(ds)):
            k=ds[i].get("original_id",ds[i].get("original_question"))
            if k in seen: continue
            seen.add(k); o.append((PROMPT.format(q=ds[i]["question"]),ln(str(ds[i]["answer"]).split("####")[-1])))
            if len(o)>=n: break
        return o

def train(a,model,tok,branches,dev):
    exs=gsm_examples("train",a.n_train)
    for n,p in model.named_parameters(): p.requires_grad_("lora_" in n)   # base frozen; lora trainable if present
    for p in branches.parameters(): p.requires_grad_(True)
    params=[p for p in model.parameters() if p.requires_grad]+list(branches.parameters())
    opt=torch.optim.AdamW(params,lr=a.lr); model.train() if a.lora else model.eval()
    step=0; steps=a.steps if a.steps>0 else (len(exs)//a.bs)*a.epochs; order=list(range(len(exs)))
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
            opt.zero_grad(); out.loss.backward(); gn=torch.nn.utils.clip_grad_norm_(params,1.0); opt.step(); step+=1
            if step%20==0 or step<=3: print(f"step {step}/{steps} loss={float(out.loss):.4f} gnorm={float(gn):.3f} gamma0={float(branches[0].gamma):.3f}",flush=True)
            if a.steps>0 and step>=a.steps: return

@torch.no_grad()
def evaluate(model,tok,dev,task,limit,isbr):
    model.eval(); test=load_test(task,limit); c=0
    for p,gold in test:
        ids=tok(p,return_tensors="pt").input_ids.to(dev)
        out=model.generate(ids,max_new_tokens=256,do_sample=False,use_cache=(not isbr),pad_token_id=tok.pad_token_id)
        if extract(tok.decode(out[0,ids.shape[1]:],skip_special_tokens=True))==gold: c+=1
    print(f"[EVAL] {task} acc={c/len(test):.4f} n={len(test)}",flush=True)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--base_model",default="meta-llama/Llama-2-7b-hf"); ap.add_argument("--r",type=int,default=64)
    ap.add_argument("--lora",action="store_true"); ap.add_argument("--lora_r",type=int,default=8)
    ap.add_argument("--n_train",type=int,default=7000); ap.add_argument("--epochs",type=int,default=3); ap.add_argument("--steps",type=int,default=0)
    ap.add_argument("--bs",type=int,default=2); ap.add_argument("--lr",type=float,default=3e-4); ap.add_argument("--cutoff",type=int,default=512)
    ap.add_argument("--seed",type=int,default=1); ap.add_argument("--out",required=True); ap.add_argument("--mode",default="train_eval"); ap.add_argument("--eval_limit",type=int,default=80)
    a=ap.parse_args(); dev='cuda'
    tok=AutoTokenizer.from_pretrained(a.base_model); tok.pad_token=tok.eos_token
    model=AutoModelForCausalLM.from_pretrained(a.base_model,torch_dtype=torch.bfloat16,device_map={"":0})
    if a.lora:
        from peft import LoraConfig, get_peft_model
        model=get_peft_model(model,LoraConfig(r=a.lora_r,lora_alpha=2*a.lora_r,lora_dropout=0.05,target_modules=["q_proj","k_proj","v_proj","up_proj","down_proj"],task_type="CAUSAL_LM"))
    branches=build(model,a.r,device=dev)
    ntr=sum(p.numel() for p in branches.parameters())+sum(p.numel() for n,p in model.named_parameters() if "lora_" in n)
    print(f"[BUILD] lora={a.lora} branch_r={a.r} trainable={ntr:,}",flush=True)
    if a.mode=="eval":
        if a.lora:
            from peft import PeftModel; base=AutoModelForCausalLM.from_pretrained(a.base_model,torch_dtype=torch.bfloat16,device_map={"":0})
            model=PeftModel.from_pretrained(base,a.out+"/lora"); branches=build(model,a.r,device=dev)
        branches.load_state_dict(torch.load(a.out+"/branches.pt",map_location=dev))
        for t in ["gsm8k","gsm_symbolic"]: evaluate(model,tok,dev,t,a.eval_limit,True); return
    if a.mode=="smoke": a.steps=5; a.n_train=64
    train(a,model,tok,branches,dev)
    os.makedirs(a.out,exist_ok=True); torch.save(branches.state_dict(),a.out+"/branches.pt")
    if a.lora: model.save_pretrained(a.out+"/lora")
    print(f"[SAVE] {a.out}",flush=True)
    if a.mode=="smoke": print(f"[SMOKE_OK] gamma0={float(branches[0].gamma):.4f}",flush=True); return
    for t in ["gsm8k","gsm_symbolic"]: evaluate(model,tok,dev,t,a.eval_limit,True)
if __name__=="__main__": main()
