import json, glob
base = {"piqa":0.729,"gsm":0.783,"squad":0.634,"mnli":0.745}  # LoRA multi-task baseline
METHODS = ["mtc_lora","mtc_aurora","mtc_glu","mtc_aurora_gate",
           "mtc_outgate_lora","mtc_outgate_aurora",
           "mtc_auroraf","mtc_auroraf_outgate",
           "mtc_ttt_lora","mtc_ttt_aurora","mtc_ttt_og_lora"]
print(f"{'method':<20} piqa   gsm    squad  mnli   AVG    (vs LoRA-mt)")
for t in METHODS:
    fs = glob.glob(f"runs/{t}/*/test_metrics_by_dataset.json") + glob.glob(f"runs/{t}/test_metrics_by_dataset.json")
    if not fs:
        print(f"{t:<20} (评估中/无结果)"); continue
    d = json.load(open(fs[0])); g = {}
    for k, v in d.items():
        for m, val in v.items():
            if "token_acc" in m:
                tt = "piqa" if "piqa" in m else "gsm" if "gsm8k" in m else "squad" if "squad" in m else "mnli"
                g[tt] = round(val, 3)
    if len(g) < 4:
        print(f"{t:<20} 部分: {g}"); continue
    avg = round(sum(g.values())/4, 3)
    dgsm = g["gsm"]-base["gsm"]; dpiqa = g["piqa"]-base["piqa"]; davg = avg-0.721
    print(f"{t:<20} {g['piqa']:<6} {g['gsm']:<6} {g['squad']:<6} {g['mnli']:<6} {avg:<6} dAVG={davg:+.3f} dgsm={dgsm:+.3f} dpiqa={dpiqa:+.3f}")
