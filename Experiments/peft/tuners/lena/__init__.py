from .config import LeNAConfig
from .model import LeNAModel

__all__ = ["LeNAConfig", "LeNAModel"]


from peft.utils import register_peft_method
from peft.utils.peft_types import PeftType


register_peft_method(
        name="lena",          # or "FLORA" in older versions
        config_cls=LeNAConfig,
        model_cls=LeNAModel,
    )