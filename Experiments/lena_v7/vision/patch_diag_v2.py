"""Patch diag_camelyon.py (Kelvin copy) to also record internal quantities for the mechanism figures."""
import sys

p = sys.argv[1]
s = open(p).read()


def rep(old, new):
    global s
    assert old in s, old[:80]
    s = s.replace(old, new, 1)


rep('''    rec = dict(s=[], F=[], theta=[], corr=[], fix=[])
    for L in layers:
        d = L.last; u = d["u"].float()                                   # (B, N, r)''',
    '''    rec = dict(s=[], F=[], p=[], theta=[], corr=[], fix=[], base=[], uhist=[], u=[], v=[])
    for L in layers:
        d = L.last; u = d["u"].float()                                   # (B, N, r)
        rec["uhist"].append(torch.stack([torch.histc(u[b].clamp(-6, 6), 60, -6, 6) for b in range(u.shape[0])]))
        rec["u"].append(u.half()); rec["v"].append(d["v"].float().half()); rec["p"].append(d["p"].float())''')
rep('''        rec["fix"].append((c0 @ Bw.T).norm(dim=-1))                       # shared-curve part''',
    '''        rec["fix"].append((c0 @ Bw.T).norm(dim=-1))                       # shared-curve part
        rec["base"].append(L._out_norm)                                   # norm of W x per token''')
rep('''    layers = lena_layers(m)
    for L in layers:
        L.log_stats = True
    with torch.autocast("cuda", dtype=torch.bfloat16):
        logit = m(x)''',
    '''    layers = lena_layers(m)
    hooks = []
    for L in layers:
        L.log_stats = True
        hooks.append(L.base.register_forward_hook(lambda mod, i, o, L=L: setattr(L, "_out_norm", o.float().norm(dim=-1))))
    with torch.autocast("cuda", dtype=torch.bfloat16):
        emb = m.tower(pixel_values=x).image_embeds.float()
        logit = m.head(F.normalize(emb, dim=-1) * 10.0).float()
    for hk in hooks:
        hk.remove()''')
rep('''    return logit.float(), {k: torch.stack(v, 1).cpu() for k, v in rec.items()}   # (B, M, ...)''',
    '''    out = {k: torch.stack(v, 1).cpu() for k, v in rec.items()}           # (B, M, ...)
    out["emb"] = emb.cpu()
    return logit.float(), out''')
rep('''            for k in ("s", "F", "theta"):
                res[k].append(r[k].half() if k == "theta" else r[k])''',
    '''            for k in ("s", "F", "p", "theta", "uhist", "emb"):
                res.setdefault(k, []).append(r[k].half() if k == "theta" else r[k])
            res.setdefault("ratio_corr", []).append(r["corr"][..., 1:].mean(-1) / r["base"][..., 1:].mean(-1))
            res.setdefault("ratio_fix", []).append(r["fix"][..., 1:].mean(-1) / r["base"][..., 1:].mean(-1))
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                res.setdefault("emb_lora", []).append(ml.tower(pixel_values=x).image_embeds.float().cpu())''')
rep('''                lm = lora_map(ml, x[:8])''',
    '''                lm = lora_map(ml, x[:8])
                with lena_v7.lena_override(mode="plain"):
                    _, rp = run_lena(m, x[:8])''')
rep('''                                            corr=r["corr"][k].numpy(), fix=r["fix"][k].numpy(), lora=lm[k].numpy()))''',
    '''                                            corr=r["corr"][k].numpy(), fix=r["fix"][k].numpy(), lora=lm[k].numpy(),
                                            u=r["u"][k].numpy(), v=r["v"][k].numpy(), theta=r["theta"][k].numpy(),
                                            theta_plain=rp["theta"][k].numpy(), s=r["s"][k].numpy(),
                                            F=r["F"][k].numpy(), p=r["p"][k].numpy()))''')
rep('''    for k in ("s", "F", "theta"):
        res[k] = torch.cat(res[k])''',
    '''    for k in ("s", "F", "p", "theta", "uhist", "emb", "emb_lora", "ratio_corr", "ratio_fix"):
        res[k] = torch.cat(res[k])''')
rep('''torch.save(res, os.path.join(a.out, "diag_cam.pt"))''', '''torch.save(res, os.path.join(a.out, a.name))''')
rep('''    for split, keep in (("validation", {0, 3, 4}), ("test", None)):''',
    '''    for split, keep in (() if a.skip_ko else (("validation", {0, 3, 4}), ("test", None))):''')
rep('''    p.add_argument("--n_ko", type=int, default=2000, help="images per split for the knock-out")''',
    '''    p.add_argument("--n_ko", type=int, default=2000, help="images per split for the knock-out")
    p.add_argument("--skip_ko", action="store_true")
    p.add_argument("--name", default="diag_cam.pt")''')
open(p, "w").write(s)
print("patched")
