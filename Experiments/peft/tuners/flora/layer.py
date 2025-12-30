from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn

from .activations import make_flora_activation
from .config import FloraConfig
from .gates import Gate


def _to_hwc(z: torch.Tensor) -> Tuple[torch.Tensor, Tuple[int, int, int], int]:
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
    """
    PEFT-compatible FloraLinear:
      - Keeps __init__(base_layer, module_key=None) (what PEFT expects)
      - Stores A/B as lora_A/lora_B so PEFT will mark them trainable
      - You can still refer to A/B if you like via aliases
      - Activation and gates are optional (can be Identity)
    """

    def __init__(self, base_layer: nn.Linear, module_key: Optional[str] = None):
        super().__init__()
        if not isinstance(base_layer, nn.Linear):
            raise TypeError(f"FloraLinear only supports nn.Linear, got {type(base_layer)}")

        self.base_layer = base_layer
        self.module_key = module_key or "<unknown>"

        # --- IMPORTANT: PEFT looks for these names for trainability ---
        self.lora_A = nn.ModuleDict()
        self.lora_B = nn.ModuleDict()

        # Optional: keep your old attribute names as aliases (same objects)
        self.A = self.lora_A
        self.B = self.lora_B

        # Activations / dropout / gates per adapter name
        self.act = nn.ModuleDict()
        self.drop = nn.ModuleDict()
        self.gate_after_a = nn.ModuleDict()
        self.gate_after_b = nn.ModuleDict()

        self.scaling: Dict[str, float] = {}
        self._active_adapter: Optional[str] = None

        # debug / logging
        self._forward_logged: Dict[str, bool] = {}
        self._dbg: Dict[str, Dict[str, bool]] = {}

    @property
    def in_features(self) -> int:
        return self.base_layer.in_features

    @property
    def out_features(self) -> int:
        return self.base_layer.out_features

    def set_active_adapter(self, name: Optional[str]):
        # PEFT will call this
        self._active_adapter = name

    def add_adapter(self, adapter_name: str, cfg: FloraConfig):
        """
        PEFT calls this (usually adapter_name == "default").
        The key fix is: put A/B into lora_A/lora_B so PEFT will unfreeze them.
        """
        r = int(cfg.r)
        if r <= 0:
            raise ValueError("FloraConfig.r must be > 0")

        # Create A/B
        A = nn.Linear(self.in_features, r, bias=False)
        B = nn.Linear(r, self.out_features, bias=False)

        # Original LoRA init style
        nn.init.kaiming_uniform_(A.weight, a=5**0.5)
        nn.init.zeros_(B.weight)

        # --- register under PEFT-recognized names ---
        self.lora_A[adapter_name] = A
        self.lora_B[adapter_name] = B

        # Make sure they are trainable (even if something else tries to freeze)
        for p in self.lora_A[adapter_name].parameters():
            p.requires_grad = True
        for p in self.lora_B[adapter_name].parameters():
            p.requires_grad = True

        # dropout
        lora_dropout = float(getattr(cfg, "lora_dropout", 0.0) or 0.0)
        self.drop[adapter_name] = nn.Dropout(p=lora_dropout) if lora_dropout > 0 else nn.Identity()

        # scaling
        self.scaling[adapter_name] = float(cfg.lora_alpha) / float(r)

        # activation (can be identity)
        self.act[adapter_name] = make_flora_activation(
            kind=cfg.flora_activation,
            mode=cfg.flora_flex_mode,
            **(cfg.flora_activation_kwargs or {}),
        )

        # gates (can be identity)
        gate_type = str(getattr(cfg, "flora_gate_type", "none")).lower()
        gate_pos = str(getattr(cfg, "flora_gate_position", "after_b")).lower()
        gate_init = float(getattr(cfg, "flora_gate_init", 1))
        init_a = 0.0 if gate_type == "rezero" else gate_init
        init_b = 0.0 if gate_type == "rezero" else gate_init
        mode = str(getattr(cfg, "flora_gate_mode", "global")).lower()
        gate_strength = str(getattr(cfg, "gate_strength", "soft")).lower()

        if gate_type != "none" and gate_pos in ("after_a", "both"):

            self.gate_after_a[adapter_name] = Gate(
                gate_type=gate_type,  # type: ignore[arg-type]
                gate_mode=mode,
                init=init_a,
                dtype=A.weight.dtype,
                device=A.weight.device,
                gate_strength=gate_strength
            )
        else:
            self.gate_after_a[adapter_name] = nn.Identity()

        if gate_type != "none" and gate_pos in ("after_b", "both"):

            self.gate_after_b[adapter_name] = Gate(
                gate_type=gate_type,  # type: ignore[arg-type]
                gate_mode=mode,
                init=init_b,
                dtype=B.weight.dtype,
                device=B.weight.device,
                gate_strength=gate_strength
            )
        else:
            self.gate_after_b[adapter_name] = nn.Identity()

        # debug flags
        self._dbg[adapter_name] = {
            "debug": bool(getattr(cfg, "flora_debug", False)),
            "verbose": bool(getattr(cfg, "flora_debug_verbose", False)),
            "forward": bool(getattr(cfg, "flora_debug_forward", False)),
            "forward_once": bool(getattr(cfg, "flora_debug_forward_once", True)),
            "check_nan": bool(getattr(cfg, "flora_debug_check_nan", False)),
        }
        self._forward_logged[adapter_name] = False

        if self._active_adapter is None:
            self._active_adapter = adapter_name

    def _pick_adapter(self, adapter_name: Optional[str]) -> Optional[str]:
        if adapter_name is not None:
            return adapter_name
        if self._active_adapter is not None:
            return self._active_adapter
        if len(self.lora_A) == 1:
            return next(iter(self.lora_A.keys()))
        return None

    def forward(self, x: torch.Tensor, adapter_name: Optional[str] = None) -> torch.Tensor:
        y = self.base_layer(x)

        name = self._pick_adapter(adapter_name)
        if name is None or name not in self.lora_A:
            return y

        A = self.lora_A[name]
        B = self.lora_B[name]
        act = self.act[name]
        drop = self.drop[name]
        gateA = self.gate_after_a[name]
        gateB = self.gate_after_b[name]
        scale = self.scaling[name]

        z = A(drop(x))
        z = gateA(z)

        if not isinstance(act, nn.Identity):
            z_hwc, _, orig_ndim = _to_hwc(z)
            z_hwc = act(z_hwc)
            z = _from_hwc(z_hwc, orig_ndim)

        dz = B(z)
        dz = gateB(dz)

        return y + dz * scale
