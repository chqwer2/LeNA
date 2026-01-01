from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Literal
from peft.utils.peft_types import PeftType

from peft.tuners.lora import LoraConfig

FloraActivation = Literal["identity", "relu", "gelu", "fourier", "spline", "polynomial"]
FloraFlexMode = Literal["global", "spatial", "channel", "voxel"]

FloraGateType = Literal["none", "sigmoid", "rezero"]
FloraGatePos = Literal["after_a", "after_b", "both"]
FloraGateMode = Literal["global", "per_dim"]


@dataclass
class FloraConfig(LoraConfig):
    # peft_type: str = field(default="FLORA", init=False)
    peft_type: PeftType = field(default=PeftType.FLORA, init=False)

    # activation
    flora_activation: FloraActivation = "identity"
    flora_activation_kwargs: Dict[str, Any] = field(default_factory=dict)
    flora_flex_mode: FloraFlexMode = "global"

    # gating
    flora_gate_type: FloraGateType = "none"
    flora_gate_position: FloraGatePos = "after_b"
    flora_gate_mode: FloraGateMode = "global"
    flora_gate_init: float = 1
    gate_strength: str = "soft"  # Literal["soft", "hard"] = "soft"

    # merge
    allow_merge: bool = False

    # ---- DEBUG ----
    flora_debug: bool = False                 # enable debug logging
    flora_debug_verbose: bool = False         # log every checked module
    flora_debug_forward: bool = False         # log forward-time execution
    flora_debug_forward_once: bool = True     # print forward log only once per module
    flora_debug_check_nan: bool = False       # warn on NaNs/Infs in adapter delta

    def __post_init__(self):
        # Let LoRA validate common fields (r, target_modules, etc.)
        super().__post_init__()
        # Then override whatever LoraConfig set
        self.peft_type = PeftType.FLORA