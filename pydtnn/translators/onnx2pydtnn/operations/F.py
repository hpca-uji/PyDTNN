# Typing related (or non important) imports
from typing import *
from pydtnn.layers.layer_and_activation_base import LayerAndActivationBase
from inspect import stack # This is only in order to get the function's name

# Functionality imports
import pydtnn.layers as layer
#import pydtnn.translators.onnx2pydtnn.constants as cons

def Flatten(info: Dict[str, Any]) -> LayerAndActivationBase:
    print(f"{stack()[0].function} args received: {info}")
    # Source: https://onnx.ai/onnx/operators/onnx__Flatten.html
    # It has one attribute (axis), but there is no equivalence in PyDTNN.
    return layer.Flatten()
# --- END Flatten --- #

def Floor(info: Dict[str, Any]) -> LayerAndActivationBase:
    print(f"{stack()[0].function} args received: {info}")
    raise NotImplementedError("Not implemented")
# --- END Floor --- #
