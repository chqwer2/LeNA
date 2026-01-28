# The way to enable LeNA in PEFT repo

1. peft/__init__.py add  "LeNAConfig, LeNAModel"
2. peft/tuners/__init__.py add  "LeNAConfig, LeNAModel"
3. Add /peft/utils/peft_types.py:PeftType LENA = "LENA"
3. create tuners/lena 
4. create under tuners/lena
    |- __init__.py
    |- activation.py  # implementation of nonlinear activation
    |- config.py      # configuration class
    |- gates.py      # implementation of gating mechanism
    |- layers.py     # implementation of LeNA layer
    |- model.py  # implementation of LeNA model wrapper


