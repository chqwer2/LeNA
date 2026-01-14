from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .activations import make_flora_activation
from .config import FloraConfig
from .gates import Gate

from copy import deepcopy

import torch
import torch.nn.functional as F
from torch import nn

from peft.utils.integrations import dequantize_module_weight, gather_params_ctx
from peft.utils.other import transpose


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

        self.norm_before_act = nn.ModuleDict()
        self.magnitude = nn.ParameterDict()
        # debug / logging
        self._forward_logged: Dict[str, bool] = {}
        self._dbg: Dict[str, Dict[str, bool]] = {}

        self.use_dora = True   # Placeholder for potential future use, no very efficiency

        w = self.base_layer.weight
        out_f, in_f = self.out_features, self.in_features

        if tuple(w.shape) == (out_f, in_f):
            self.fan_in_fan_out = True  # standard nn.Linear
        elif tuple(w.shape) == (in_f, out_f):
            self.fan_in_fan_out = False  # transposed storage
        else:
            # Fallback: default to False but warn loudly
            self.fan_in_fan_out = True

        self.init = False

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
        # nn.init.kaiming_uniform_(A.weight, a=5**0.5, nonlinearity='leaky_relu')
        nn.init.xavier_uniform_(A.weight)
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
        # This is for LoRA
        # self.scaling[adapter_name] = float(cfg.lora_alpha) / float(r)

        flora_nonlinear_scale = 1.0
        self.scaling[adapter_name] = float(cfg.lora_alpha) / float(r) * flora_nonlinear_scale

        self.norm_before_act[adapter_name] = nn.LayerNorm(r)

        # activation (can be identity)
        self.act[adapter_name] = make_flora_activation(
            kind=cfg.flora_activation,
            mode=cfg.flora_flex_mode,
            **(cfg.flora_activation_kwargs or {}),
        )

        # weight_norm

        with torch.no_grad():
            # Compute effective LoRA weight matrix: B @ A
            lora_weight = B.weight @ A.weight  # [out_features, in_features]

            # Get base layer weight
            weight = dequantize_module_weight(self.base_layer)
            lora_weight = lora_weight.to(device=weight.device, dtype=weight.dtype)

            # Compute initial magnitude as the column-wise norm of (W + scaling * ΔW)
            weight_norm = self.get_weight_norm(weight, lora_weight, self.scaling[adapter_name])

            # Initialize magnitude parameter with proper shape [out_features]
            self.magnitude[adapter_name] = nn.Parameter(
                weight_norm.clone(),
                requires_grad=True
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

    def get_weight_norm(self, weight, lora_weight, scaling) -> torch.Tensor:
        # calculate L2 norm of weight matrix, column-wise
        # weight = transpose(weight, self.fan_in_fan_out)
        # lora_weight = transpose(lora_weight, self.fan_in_fan_out)

        if lora_weight.shape != weight.shape:
            lora_weight = transpose(lora_weight, True)

        # print("Wright=", lora_weight.shape, weight.shape)

        weight = weight + scaling * lora_weight
        weight_norm = torch.linalg.norm(weight, dim=1).to(weight.dtype)
        return weight_norm

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

        norm = self.norm_before_act[name]

        if self.use_dora:
            # x_eye = torch.eye(A.weight.shape[1], device=A.weight.device, dtype=x.dtype)
            x_eye = torch.eye(A.weight.shape[1], device=A.weight.device, dtype=x.dtype)

            # if not isinstance(act, nn.Identity) and self.init:
            #     z_hwc, _, orig_ndim = _to_hwc(x_eye)
            #     z_hwc = act(z_hwc)
            #     x_eye = _from_hwc(z_hwc, orig_ndim)

            lora_weight = B(A(x_eye))

            weight = dequantize_module_weight(self.base_layer)
            weight = weight.to(x.dtype)
            weight_norm = self.get_weight_norm(weight, lora_weight.detach(), scale)

            magnitude = self.magnitude[name]

            weight_norm = weight_norm.detach()
            mag_norm_scale = (magnitude / weight_norm).view(1, -1)

        # z = A(drop(x))
        z = A(x)  #
        z = gateA(z)

        if not isinstance(act, nn.Identity):
            self.init = True
            z_hwc, _, orig_ndim = _to_hwc(z)
            z_hwc = act(z_hwc)
            z = _from_hwc(z_hwc, orig_ndim)

            z = z.clamp(-10.0, 10.0)

        dz = B(z)
        dz = gateB(dz)

        # if hasattr(self, 'magnitude') and name in self.magnitude:
        #     dz_norm = dz.norm(p=2, dim=-1, keepdim=True).detach() + 1e-8
        #     dz = self.magnitude[name] * (dz / dz_norm)

        if self.use_dora:
            if self.base_layer.bias is not None:
                y = y - self.base_layer.bias

            # print("mag_norm_scale = ", mag_norm_scale.max(), mag_norm_scale.min())

            return  mag_norm_scale  * y + mag_norm_scale * dz * scale

        dz = self.magnitude[name] * dz

        return y + dz * scale


