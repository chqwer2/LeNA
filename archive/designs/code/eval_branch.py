import torch, importlib.util, re, argparse
spec=importlib.util.spec_from_file_location("tb","ttt_branch.py"); tb=importlib.util.module_from_spec(spec); spec.loader.exec_module(tb)
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from datasets import load_dataset

BM="meta-llama/Llama-2-7b-hf"; DEV='cuda'
tok=AutoTokenizer.from_pretrained(BM); tok.pad_token=tok.eos_token
P=tb.PROMPT

def ln(s):
    n=re.findall(r"-?\d[\d,]*\.?\d*", str(s).replace(",","")); return n[-1].rstrip(".") if n else None
def extract(gen):
    # FAIR: first '#### <number>' (robust to LoRA's repeated-#### degeneration), else last number
    m=re.search(r"####\s*\$?\s*(-?[\d][\d,]*\.?\d*)", gen)
    if m: return m.group(1).replace(",","").rstrip(".")
    return ln(gen)

def get_data(task, n):
    if task=="gsm8k":
        ds=load_dataset("openai/gsm8k","main",split="test")
        return [(P.format(q=ds[i]["question"]), ln(str(ds[i]["answer"]).split("####")[-1])) for i in range(min(n,len(ds)))]
    if task=="gsm_symbolic":   # DIVERSE: 1 instance per template
        ds=load_dataset("apple/GSM-Symbolic","main",split="test")
        seen=set(); out=[]
        for i in range(len(ds)):
            k=ds[i].get("original_id", ds[i].get("original_question"))
            if k in seen: continue
            seen.add(k); out.append((P.format(q=ds[i]["question"]), ln(str(ds[i]["answer"]).split("####")[-1])))
            if len(out)>=n: break
        return out
    if task=="svamp":
        ds=load_dataset("ChilleD/SVAMP",split="test")
        return [(P.format(q=(str(ds[i]["Body"]).strip()+" "+str(ds[i]["Question"]).strip())), ln(str(ds[i]["Answer"]))) for i in range(min(n,len(ds)))]
    if task=="asdiv":
        ds=load_dataset("EleutherAI/asdiv",split="validation")
        return [(P.format(q=(str(ds[i]["body"]).strip()+" "+str(ds[i]["question"]).strip())), ln(str(ds[i]["answer"]))) for i in range(min(n,len(ds)))]

@torch.no_grad()
def run(model, is_branch, tag, n):
    model.eval()
    for task in ["gsm8k","gsm_symbolic","svamp","asdiv"]:
        data=get_data(task,n); c=0
        for p,gold in data:
            ids=tok(p,return_tensors="pt").input_ids.to(DEV)
            out=model.generate(ids, max_new_tokens=256, do_sample=False, use_cache=(not is_branch), pad_token_id=tok.pad_token_id)
            if extract(tok.decode(out[0,ids.shape[1]:], skip_special_tokens=True))==gold: c+=1
        print(f"[OOD] {tag:14s} {task:13s} acc={c/len(data):.4f} n={len(data)}", flush=True)

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--which",required=True); ap.add_argument("--n",type=int,default=80); a=ap.parse_args()
    m=AutoModelForCausalLM.from_pretrained(BM,torch_dtype=torch.bfloat16,device_map={"":0}); isbr=False
    if a.which=="base": pass
    elif a.which.startswith("lora_"):
        s=a.which.split("_")[1]; m=PeftModel.from_pretrained(m, f"runs/r8_lora_{s}/lora")
    elif a.which.startswith("branch_"):
        s=a.which.split("_")[1]; br=tb.build(m,64,device=DEV); br.load_state_dict(torch.load(f"runs/ttbranch_r64_{s}/branches.pt",map_location=DEV)); isbr=True
    run(m, isbr, a.which, a.n)
    print("EVALBRANCH_DONE", a.which, flush=True)
if __name__=="__main__": main()
