import ast
F = "peft/tuners/lena/layer.py"
s = open(F).read()

# 1) SteerHead.__init__: accept init_U (SVD directions)
old_init = '''    def __init__(self, d_out, r, p=4, k=1):
        super().__init__()
        self.k=int(k)
        self.U=nn.Parameter(torch.randn(d_out,int(p))/(float(d_out)**0.5))'''
new_init = '''    def __init__(self, d_out, r, p=4, k=1, init_U=None):
        super().__init__()
        self.k=int(k)
        if init_U is not None:
            U0=torch.zeros(d_out,int(p))
            c=min(int(p), init_U.shape[1]); U0[:, :c]=init_U[:, :c].float()
            self.U=nn.Parameter(U0)                      # SVD-init: weight-aligned directions
        else:
            self.U=nn.Parameter(torch.randn(d_out,int(p))/(float(d_out)**0.5))'''
assert old_init in s, "SteerHead __init__ anchor missing"
if "init_U=None" not in s:
    s = s.replace(old_init, new_init, 1); print("Fix1: SteerHead accepts init_U")
else:
    print("Fix1: already present")

# 2) layer setup: compute SVD directions when LENA_STEER_SVD set
old_create = '''            self.steer[adapter_name] = SteerHead(
                self.out_features, r,
                p=int(os.environ.get("LENA_STEER_P", "4")),
                k=int(os.environ.get("LENA_STEER_K", "1")),
            )'''
new_create = '''            _p=int(os.environ.get("LENA_STEER_P", "4"))
            _initU=None
            if os.environ.get("LENA_STEER_SVD"):
                try:
                    _W=self.base_layer.weight.data.float()
                    _q=min(_p+4, min(_W.shape)-1)
                    _U,_S,_V=torch.svd_lowrank(_W, q=_q)
                    _initU=_U[:, :_p].contiguous()          # top-p left singular vectors
                except Exception as _e:
                    _initU=None
            self.steer[adapter_name] = SteerHead(
                self.out_features, r, p=_p,
                k=int(os.environ.get("LENA_STEER_K", "1")),
                init_U=_initU,
            )'''
assert old_create in s, "steer create anchor missing"
if "LENA_STEER_SVD" not in s:
    s = s.replace(old_create, new_create, 1); print("Fix2: SVD-init wired in setup")
else:
    print("Fix2: already present")

open(F, "w").write(s)
ast.parse(open(F).read())
print("syntax OK")
