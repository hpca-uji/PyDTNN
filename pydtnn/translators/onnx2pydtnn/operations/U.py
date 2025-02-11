# Typing related (or non important) imports
from typing import *
from pydtnn.layers.layer_and_activation_base import LayerAndActivationBase
from inspect import stack # This is only in order to get the function's name

# Functionality imports
from constants import CONST_ATTRIBUTES

def Unique(info: Dict[str, Any]) -> LayerAndActivationBase:
    print(f"{stack()[0].function()} args received: {info}")
    raise NotImplementedError("Not implemented")
# --- END Unique --- #

def Unsqueeze(info: Dict[str, Any]) -> LayerAndActivationBase:
    # Onnx information: https://onnx.ai/onnx/operators/onnx__Unsqueeze.html
    print(f"{stack()[0].function()} args received: {info}")
    ONNX_AXES = "axes"
    dict_attributes = info[CONST_ATTRIBUTES]
    # TODO
    args = {}

    if ONNX_AXES in dict_attributes:
        args["¡¡¡TODO: PUT THE ACTUAL NAME!!!"] = dict_attributes[ONNX_AXES]
    # return the unsqueeze class.
    raise NotImplementedError("Not implemented")
    
# --- END Unsqueeze --- #

def Upsample(info: Dict[str, Any]) -> LayerAndActivationBase:
    print(f"{stack()[0].function()} args received: {info}")
    raise NotImplementedError("Not implemented")
# --- END Upsample --- #
