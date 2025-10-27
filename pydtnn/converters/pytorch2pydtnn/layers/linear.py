# _______________________________________________________________________________________________________________
# In this file must be implemented only the translation of PyTorch Linear layers to its PyDTNN equivalent.
# _______________________________________________________________________________________________________________

# Typing related (or non important) imports
from typing import Dict, Any

# Functionality imports
from pydtnn.layers.fc import FC
import pydtnn.converters.pytorch2pydtnn.common as cm


# ------------------ #
def Linear(args: Dict[str, Any]) -> FC:
    # https://pytorch.org/docs/stable/generated/torch.nn.Linear.html#torch.nn.Linear

    # PyTorch attributes:
    # Not used: in_features (It's not used due the way the layer's initialization works in PyDTNN)
    PYTORCH_BIAS = "bias"
    PYTORCH_OUT_FEATURES = "out_features"
    torch_dict_keys = [PYTORCH_BIAS, PYTORCH_OUT_FEATURES]
    # ---- #

    # PyDTNN attributes:
    # Not used: activation
    # Used, but in a different place: weights_initializer, biases_initializer
    PYDTNN_BIAS = "use_bias"
    PYDTNN_SHAPE = "shape"
    pydtnn_dict_keys = [PYDTNN_BIAS, PYDTNN_SHAPE]
    # ---- #

    layer_args = cm.prepare_pydtnn_arguments(arguments=args[cm.ARGUMENTS], torch_dict_keys=torch_dict_keys, pydtnn_dict_keys=pydtnn_dict_keys)

    # PyDTNN expects the shape as a tuple instead of an int.
    if PYDTNN_SHAPE in layer_args and isinstance(layer_args[PYDTNN_SHAPE], int):
        layer_args[PYDTNN_SHAPE] = (layer_args[PYDTNN_SHAPE], )

    return FC(**layer_args)
