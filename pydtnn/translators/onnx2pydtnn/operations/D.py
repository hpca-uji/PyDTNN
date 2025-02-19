# Typing related (or non important) imports
from typing import *
from pydtnn.layers.layer_and_activation_base import LayerAndActivationBase
from inspect import stack # This is only in order to get the function's name

# Functionality imports
import pydtnn.layers as layer
import pydtnn.translators.onnx2pydtnn.constants as cons

def DFT(info: Dict[str, Any]) -> LayerAndActivationBase:
    print(f"Operation: {stack()[0].function}\nargs received: {info}")
    raise NotImplementedError("Not implemented")
# --- END DFT --- #

def DeformConv(info: Dict[str, Any]) -> LayerAndActivationBase:
    print(f"Operation: {stack()[0].function}\nargs received: {info}")
    raise NotImplementedError("Not implemented")
# --- END DeformConv --- #

def DepthToSpace(info: Dict[str, Any]) -> LayerAndActivationBase:
    print(f"Operation: {stack()[0].function}\nargs received: {info}")
    raise NotImplementedError("Not implemented")
# --- END DepthToSpace --- #

def DequantizeLinear(info: Dict[str, Any]) -> LayerAndActivationBase:
    print(f"Operation: {stack()[0].function}\nargs received: {info}")
    raise NotImplementedError("Not implemented")
# --- END DequantizeLinear --- #

def Det(info: Dict[str, Any]) -> LayerAndActivationBase:
    print(f"Operation: {stack()[0].function}\nargs received: {info}")
    raise NotImplementedError("Not implemented")
# --- END Det --- #

def Div(info: Dict[str, Any]) -> LayerAndActivationBase:
    print(f"Operation: {stack()[0].function}\nargs received: {info}")
    raise NotImplementedError("Not implemented")
# --- END Div --- #

def Dropout(info: Dict[str, Any]) -> LayerAndActivationBase:
    print(f"Operation: {stack()[0].function}")
    print(f"attributes: {info[cons.CONST_ATTRIBUTES]}")
    # Onnx attributes names from: https://onnx.ai/onnx/operators/onnx__Dropout.html#l-onnx-doc-dropout
    ONNX_SEED = "seed" # TODO: Check if the random seed it's important. If it is, check how to set it.
    # PyDTNN attributes names from Dropout class.
    PYDTNN_RATE = "rate"

    args = {}

    # TODO: Check if this is correct.
    # Droput can receive 3 inputs: the previous layer output [Tensor], 
    #   the ratio (of random dropout) [Float] and if it's in training mode [bool]
    # Then if it has more than one input and it's not a bool or the previous layer output, it is the ratio.
    _other_inputs = set(enumerate(info[cons.CONST_ALL_INPUTS])) - set(enumerate(info[cons.CONST_INPUTS]))
    other_inputs = [elem[1] for elem in sorted(_other_inputs, key=lambda x: x[0])]
        
    if len(other_inputs) > 0: 
        for k in other_inputs:
            elem = info[cons.CONST_WEIGHTS][other_inputs[k]]
            if not isinstance(elem, bool):
                args[PYDTNN_RATE] = elem
                break

    print(f"==> args: {args}")

    return layer.Dropout(**args)
# --- END Dropout --- #

def DynamicQuantizeLinear(info: Dict[str, Any]) -> LayerAndActivationBase:
    print(f"Operation: {stack()[0].function}\nargs received: {info}")
    raise NotImplementedError("Not implemented")
# --- END DynamicQuantizeLinear --- #
