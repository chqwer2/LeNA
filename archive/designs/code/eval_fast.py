import os, re, json, argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

PROMPT="### Task: Solve the math problem. End with '#### <number>'.\n\nQuestion: {q}\n\nAnswer:"

def ln(s):
    n=re.findall(r"-?\d[\d,]*\.?\d*", str(s).replace(",",""))
    return n[-1].rstrip(".") if n else None
def norm(x):
    if x is None: return None
    x=str(x).replace(",","").strip().rstrip(".")
    return x
def extract(g):
    m=re.search(r"####\s*\$?\s*(-?[\d][\d,]*\.?\d*)", g)   # FIRST #### number (fair; survives repetition)
    return norm(m.group(1)) if m else norm(ln(g))

def load_task(task, limit):
    if task=="gsm8k":
        ds=load_dataset("openai/gsm8k","main",split="test")
        return [(PROMPT.format(q=ds[i]["question"]), norm(ln(str(ds[i]["answer"]).split("####")[-1]))) for i in range(min(limit,len(ds)))]
    if task=="gsm_symbolic":
        ds=load_dataset("apple/GSM-Symbolic","main",split="test"); seen=set(); o=[]
        for i in range(len(ds)):
            k=ds[i].get("original_id", ds[i].get("original_question"))
            if k in seen: continue
            seen.add(k); o.append((PROMPT.format(q=ds[i]["question"]), norm(ln(str(ds[i]["answer"]).split("####")[-1]))))
            if len(o)>=limit: break
        return o
    if task=="svamp":
        ds=load_dataset("ChilleD/SVAMP",split="test")
        return [(PROMPT.format(q=(ds[i]["Body"].strip()+" "+ds[i]["Question"].strip())), norm(ds[i]["Answer"])) for i in range(min(limit,len(ds)))]
    if task=="asdiv":
        ds=load_dataset("EleutherAI/asdiv",split="validation")
        return [(PROMPT.format(q=(ds[i]["body"].strip()+" "+ds[i]["question"].strip())), norm(ln(str(ds[i]["answer"])))) for i in range(min(limit,len(ds)))]
    raise ValueError(task)

@torch.no_grad()
def run(model, tok, data, bs, maxtok, dev):
    c=0
    for i in range(0,len(data),bs):
        batch=data[i:i+bs]; prompts=[p for p,_ in batch]; golds=[g for _,g in batch]
        enc=tok(prompts, return_tensors="pt", padding=True, truncation=True, max_length=768).to(dev)
        out=model.generate(**enc, max_new_tokens=maxtok, do_sample=False, pad_token_id=tok.pad_token_id)
        gen=out[:, enc.input_ids.shape[1]:]
        for j in range(len(batch)):
            if extract(tok.decode(gen[j], skip_special_tokens=True))==golds[j]: c+=1
    return c/max(len(data),1)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--base_model",default="meta-llama/Llama-2-7b-hf")
    ap.add_argument("--adapter",default=None); ap.add_argument("--base_only",action="store_true")
    ap.add_argument("--tasks",default="gsm8k"); ap.add_argument("--limit",type=int,default=500)
    ap.add_argument("--bs",type=int,default=16); ap.add_argument("--maxtok",type=int,default=128)
    ap.add_argument("--out",default=None)
    a=ap.parse_args()
    cache=os.path.join(os.environ["HF_HOME"],"hub") if os.environ.get("HF_HOME") else None
    tok=AutoTokenizer.from_pretrained(a.base_model, cache_dir=cache)
    if tok.pad_token is None: tok.pad_token=tok.eos_token
    tok.padding_side="left"
    model=AutoModelForCausalLM.from_pretrained(a.base_model, dtype=torch.float16, device_map={"":0}, cache_dir=cache)
    if a.adapter and not a.base_only:
        from peft import PeftModel; model=PeftModel.from_pretrained(model, a.adapter); print(f"[eval-fast] adapter {a.adapter}",flush=True)
    else:
        print("[eval-fast] base",flush=True)
    model.eval(); dev=next(model.parameters()).device
    res={}
    for t in [x.strip() for x in a.tasks.split(",") if x.strip()]:
        acc=run(model, tok, load_task(t,a.limit), a.bs, a.maxtok, dev)
        res[t]={"exact_match":acc}; print(f"[FAST] {t} exact_match={acc:.4f}",flush=True)
    if a.out: json.dump(res, open(a.out,"w"), indent=2); print(f"[WROTE] {a.out}",flush=True)
if __name__=="__main__": main()
