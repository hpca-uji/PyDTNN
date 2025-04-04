# _______________________________________________________________________________________________________________
# In this file must be implemented only the translation of PyTorch Activations layers to its PyDTNN equivalent.
# _______________________________________________________________________________________________________________

# Typing related (or non important) imports
from typing import *
from pydtnn.layers.layer_and_activation_base import LayerAndActivationBase
from inspect import stack # This is only in order to get the function's name

# Functionality imports
from pydtnn import activations
import pydtnn.translators.pytorch2pydtnn.common as cm
# ------------------- #

def ReLU(args: Dict[str, Any]) -> activations.Relu:
    # https://pytorch.org/docs/stable/generated/torch.nn.ReLU.html#torch.nn.ReLU
    # Not used Pytorch's parameters: inplace.

    not_used =args
    return activations.Relu()
# --- END AdaptiveAvgPool2d --- #