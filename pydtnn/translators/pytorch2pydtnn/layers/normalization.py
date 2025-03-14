# _______________________________________________________________________________________________________________
# In this file must be implemented only the translation of PyTorch Convolutional layers to its PyDTNN equivalent.
# _______________________________________________________________________________________________________________

# Typing related (or non important) imports
from typing import *
from inspect import stack # This is only in order to get the function's name

# Functionality imports
from pydtnn import layers
import pydtnn.translators.pytorch2pydtnn.constats as cons

# ------------------ #
def BatchNorm2d(args: Dict[str, Any]) -> layers.BatchNormalization:
    # https://pytorch.org/docs/stable/generated/torch.nn.BatchNorm2d.html#torch.nn.BatchNorm2d

    print(f"Layer: {stack()[0].function}")
    # PyTorch attributes:
    # Not used: num_features, affine, track_running_stats
    PYTORCH_EPS = "eps" #Float
    PYTORCH_MOMENTUM = "momentum" # Float

    torch_dict_keys = [PYTORCH_MOMENTUM, PYTORCH_EPS]
    # ---- #

    # PyDTNN attributes:
    # Not used: beta, gamma
    PYDTNN_MOMENTUM = "momentum"
    PYDTNN_EPSILON = "epsilon"

    pydtnn_dict_keys = [PYDTNN_MOMENTUM, PYDTNN_EPSILON]
    # ---- #   

    layer_args = cons.prepare_pydtnn_arguments(arguments = args[cons.ARGUMENTS], torch_dict_keys = torch_dict_keys, pydtnn_dict_keys = pydtnn_dict_keys)

    cons.print_dict(args[cons.ARGUMENTS], "args[cons.ARGUMENTS]")
    cons.print_dict(layer_args, "layer_args")

    return layers.BatchNormalization(**layer_args)
# --- END BatchNorm2d --- #