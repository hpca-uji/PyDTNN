# Typing related (or non important) imports
from typing import *
from pydtnn.layers.layer_and_activation_base import LayerAndActivationBase
from inspect import stack # This is only in order to get the function's name

# Functionality imports
import pydtnn.layers as layer
import pydtnn.translators.onnx2pydtnn.constants as cons

def BatchNormalization(info: Dict[str, Any]) -> LayerAndActivationBase:
    print(f"Operation: {stack()[0].function}\nargs received: {info}")

    # Onnx attributes names from: https://onnx.ai/onnx/operators/onnx__BatchNormalization.html#l-onnx-doc-batchnormalization
    ONNX_EPSILON = "epsilon"
    ONNX_MOMENTUM = "momentum" 
    # PyDTNN attributes names from BatchNormalization class.
    PYDTNN_EPSILON = "epsilon"
    PYDTNN_MOMENTUM = "momentum"
    
    args = dict()
    dict_attributes = info[cons.CONST_ATTRIBUTES]

    if ONNX_EPSILON in dict_attributes: 
        args[PYDTNN_EPSILON] = dict_attributes[ONNX_EPSILON]
    if ONNX_MOMENTUM in dict_attributes: 
        args[PYDTNN_MOMENTUM] = dict_attributes[ONNX_MOMENTUM]

    return layer.BatchNormalization(*args)    
# --- END BatchNormalization --- #

def Bernoulli(info: Dict[str, Any]) -> LayerAndActivationBase:
    print(f"Operation: {stack()[0].function}\nargs received: {info}")
    raise NotImplementedError("Not implemented")
# --- END Bernoulli --- #

def BitShift(info: Dict[str, Any]) -> LayerAndActivationBase:
    print(f"Operation: {stack()[0].function}\nargs received: {info}")
    raise NotImplementedError("Not implemented")
# --- END BitShift --- #

def BitwiseAnd(info: Dict[str, Any]) -> LayerAndActivationBase:
    print(f"Operation: {stack()[0].function}\nargs received: {info}")
    raise NotImplementedError("Not implemented")
# --- END BitwiseAnd --- #

def BitwiseNot(info: Dict[str, Any]) -> LayerAndActivationBase:
    print(f"Operation: {stack()[0].function}\nargs received: {info}")
    raise NotImplementedError("Not implemented")
# --- END BitwiseNot --- #

def BitwiseOr(info: Dict[str, Any]) -> LayerAndActivationBase:
    print(f"Operation: {stack()[0].function}\nargs received: {info}")
    raise NotImplementedError("Not implemented")
# --- END BitwiseOr --- #

def BitwiseXor(info: Dict[str, Any]) -> LayerAndActivationBase:
    print(f"Operation: {stack()[0].function}\nargs received: {info}")
    raise NotImplementedError("Not implemented")
# --- END BitwiseXor --- #

def BlackmanWindow(info: Dict[str, Any]) -> LayerAndActivationBase:
    print(f"Operation: {stack()[0].function}\nargs received: {info}")
    raise NotImplementedError("Not implemented")
# --- END BlackmanWindow --- #
