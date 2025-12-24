from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn

from .activations import make_flora_activation
from .config import FloraConfig
from .gates import Gate


def _to_hwc(z: torch.Tensor) -> Tuple[torch.Tensor, Tuple[int, int, int], int]:
    """
    Make z look like [...,H,W,C] where C is last dim.
    Return (z_hwc, (H,W,C), orig_ndim).
    """
    orig_ndim = z.ndim
    if z.ndim >= 4:
        H, W, C = int(z.shape[-3]), int(z.shape[-2]), int(z.shape[-1])
        return z, (H, W, C), orig_ndim
    if z.ndim == 3:
        H, C = int(z.shape[-2]), int(z.shape[-1])
        return z.unsqueeze(-2), (H, 1, C), orig_ndim
    if z.ndim == 2:
        C = int(z.shape[-1])
        return z.unsqueeze(-2).unsqueeze(-2), (1, 1, C), orig_ndim
    raise ValueError(f"Unsupported z shape: {tuple(z.shape)}")


def _from_hwc(z_hwc: torch.Tensor, orig_ndim: int) -> torch.Tensor:
    if orig_ndim >= 4:
        return z_hwc
    if orig_ndim == 3:
        return z_hwc.squeeze(-2)
    if orig_ndim == 2:
        return z_hwc.squeeze(-2).squeeze(-2)
    raise ValueError("orig_ndim must be >=2")


class FloraLinear(nn.Module):
    def __init__(self, base_layer: nn.Linear, module_key: Optional[str] = None):
        super().__init__()
        if not isinstance(base_layer, nn.Linear):
            raise TypeError(f"FloraLinear only supports nn.Linear, got {type(base_layer)}")
        self.base_layer = base_layer

        # Module name/path for debug messages
        self.module_key = module_key or "<unknown>"

        self.A = nn.ModuleDict()
        self.B = nn.ModuleDict()
        self.act = nn.ModuleDict()
        self.drop = nn.ModuleDict()

        self.gate_after_a = nn.ModuleDict()
        self.gate_after_b = nn.ModuleDict()

        self.scaling: Dict[str, float] = {}
        self._active_adapter: Optional[str] = None

        # forward debug "print once"
        self._forward_logged: Dict[str, bool] = {}

        # store debug flags per adapter
        self._dbg: Dict[str, Dict[str, bool]] = {}

    @property
    def in_features(self) -> int:
        return self.base_layer.in_features

    @property
    def out_features(self) -> int:
        return self.base_layer.out_features

    def set_active_adapter(self, name: Optional[str]):
        self._active_adapter = name

    def add_adapter(self, adapter_name: str, cfg: FloraConfig):
        r = int(cfg.r)
        if r <= 0:
            raise ValueError("FloraConfig.r must be > 0")

        self.A[adapter_name] = nn.Linear(self.in_features, r, bias=False)
        self.B[adapter_name] = nn.Linear(r, self.out_features, bias=False)

        nn.init.kaiming_uniform_(self.A[adapter_name].weight, a=5**0.5)
        nn.init.zeros_(self.B[adapter_name].weight)

        self.drop[adapter_name] = (
            nn.Dropout(p=cfg.lora_dropout) if getattr(cfg, "lora_dropout", 0.0) and cfg.lora_dropout > 0 else nn.Identity()
        )
        self.scaling[adapter_name] = float(cfg.lora_alpha) / float(r)

        self.act[adapter_name] = make_flora_activation(
            kind=cfg.flora_activation,
            mode=cfg.flora_flex_mode,
            **(cfg.flora_activation_kwargs or {}),
        )

        # Gates
        gate_type = str(getattr(cfg, "flora_gate_type", "none")).lower()
        gate_pos = getattr(cfg, "flora_gate_position", "after_b")
        gate_init = float(getattr(cfg, "flora_gate_init", -6.0))
        init_a = 0.0 if gate_type == "rezero" else gate_init
        init_b = 0.0 if gate_type == "rezero" else gate_init

        if gate_type != "none" and gate_pos in ("after_a", "both"):
            mode_a = getattr(cfg, "flora_gate_mode_after_a", "global")
            self.gate_after_a[adapter_name] = Gate(
                gate_type=gate_type,  # type: ignore[arg-type]
                gate_mode=mode_a,
                n_features=r if mode_a == "per_dim" else None,
                init=init_a,
                dtype=self.A[adapter_name].weight.dtype,
                device=self.A[adapter_name].weight.device,
            )
        else:
            self.gate_after_a[adapter_name] = nn.Identity()

        if gate_type != "none" and gate_pos in ("after_b", "both"):
            mode_b = getattr(cfg, "flora_gate_mode_after_b", "global")
            self.gate_after_b[adapter_name] = Gate(
                gate_type=gate_type,  # type: ignore[arg-type]
                gate_mode=mode_b,
                n_features=self.out_features if mode_b == "per_dim" else None,
                init=init_b,
                dtype=self.B[adapter_name].weight.dtype,
                device=self.B[adapter_name].weight.device,
            )
        else:
            self.gate_after_b[adapter_name] = nn.Identity()

        # Debug flags
        self._dbg[adapter_name] = {
            "debug": bool(getattr(cfg, "flora_debug", False)),
            "verbose": bool(getattr(cfg, "flora_debug_verbose", False)),
            "forward": bool(getattr(cfg, "flora_debug_forward", False)),
            "forward_once": bool(getattr(cfg, "flora_debug_forward_once", True)),
            "check_nan": bool(getattr(cfg, "flora_debug_check_nan", False)),
        }
        self._forward_logged[adapter_name] = False

    def _get_adapter(self, adapter_name: Optional[str]) -> Optional[str]:
        # 1) explicit override
        if adapter_name is not None:
            return adapter_name

        # 2) active adapter set by model wrapper
        if self._active_adapter is not None:
            return self._active_adapter

        # 3) fallback: if there's exactly one adapter in the module, use it
        if len(self.A) == 1:
            return next(iter(self.A.keys()))

        return None

    def forward(self, x: torch.Tensor, adapter_name: Optional[str] = None) -> torch.Tensor:

        y = self.base_layer(x)
        name = self._get_adapter(adapter_name)
        if name is None or name not in self.A:
            # print("Returning base layer output without Flora adapter")
            return y



        A = self.A[name]
        B = self.B[name]
        act = self.act[name]
        drop = self.drop[name]
        gateA = self.gate_after_a[name]
        gateB = self.gate_after_b[name]
        scale = self.scaling[name]

        dbg = self._dbg.get(name, {})
        do_fwd_log = dbg.get("forward", False)
        once = dbg.get("forward_once", True)


        z = A(drop(x))
        z = gateA(z)

        used_activation = not isinstance(act, nn.Identity)
        if used_activation:
            z_hwc, (H, W, C), orig_ndim = _to_hwc(z)
            z_hwc = act(z_hwc)
            z = _from_hwc(z_hwc, orig_ndim)
        else:
            H = W = C = -1  # for printing

        dz = B(z)
        dz = gateB(dz)
        out = y + dz * scale

        if do_fwd_log and (not once or not self._forward_logged.get(name, False)):
            self._forward_logged[name] = True
            gate_type = type(gateA).__name__ if not isinstance(gateA, nn.Identity) else "OFF"
            gate_type_b = type(gateB).__name__ if not isinstance(gateB, nn.Identity) else "OFF"
            act_name = type(act).__name__ if used_activation else "OFF"
            print(
                f"[FLORA:FWD] {self.module_key} adapter='{name}' "
                f"A_out={tuple(z.shape)} act={act_name} (H,W,C={H},{W},{C}) "
                f"gateA={gate_type} gateB={gate_type_b} scale={scale:.4g}"
            )

        if dbg.get("check_nan", False):
            if not torch.isfinite(dz).all():
                print(f"[FLORA:WARN] Non-finite adapter delta in {self.module_key} adapter='{name}'")

        return out
