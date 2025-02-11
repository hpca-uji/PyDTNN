# Typing related (or non important) imports
from typing import *
from pydtnn.layers.layer_and_activation_base import LayerAndActivationBase
from inspect import stack # This is only in order to get the function's name

# Functionality imports
import pydtnn.layers as layer
from constants import CONST_INPUTS, CONST_ATTRIBUTES, CONST_PREV_LAYERS

def GRU(info: Dict[str, Any]) -> LayerAndActivationBase:
    print(f"{stack()[0].function()} args received: {info}")
    raise NotImplementedError("Not implemented")
# --- END GRU --- #

def Gather(info: Dict[str, Any]) -> LayerAndActivationBase:
    print(f"{stack()[0].function()} args received: {info}")
    raise NotImplementedError("Not implemented")
# --- END Gather --- #

def GatherElements(info: Dict[str, Any]) -> LayerAndActivationBase:
    print(f"{stack()[0].function()} args received: {info}")
    raise NotImplementedError("Not implemented")
# --- END GatherElements --- #

def GatherND(info: Dict[str, Any]) -> LayerAndActivationBase:
    print(f"{stack()[0].function()} args received: {info}")
    raise NotImplementedError("Not implemented")
# --- END GatherND --- #

def Gelu(info: Dict[str, Any]) -> LayerAndActivationBase:
    print(f"{stack()[0].function()} args received: {info}")
    raise NotImplementedError("Not implemented")
# --- END Gelu --- #

def Gemm(info: Dict[str, Any]) -> LayerAndActivationBase:
    print(f"{stack()[0].function()} args received: {info}")
    raise NotImplementedError("Not implemented")
    # FC,   # TODO: Revisar
# --- END Gemm --- #

def GlobalAveragePool(info: Dict[str, Any]) -> LayerAndActivationBase:
    print(f"{stack()[0].function()} args received: {info}")
    # 1.- Onnx Information: https://onnx.ai/onnx/operators/onnx__GlobalAveragePool.html
    
    # PyDTNN attributes names from AbstractPool2DLayer class.
    PYDTNN_POOL_SHAPE = "pool_shape"
    PYDTNN_STRIDE = "stride"

    print(f"{stack()[0].function()} args received: {info}")
    args = dict()

    operations = info[CONST_PREV_LAYERS]
    _input = info[CONST_INPUTS][0] # It should be a list with only one input

    # TODO: check if this is correct.

    # "This is equivalent to AveragePool with kernel size equal to the spatial dimension of input tensor." [1]
    args[PYDTNN_POOL_SHAPE] = operations[_input].shape
    args[PYDTNN_STRIDE] = 1
    
    return layer.AveragePool2D(*args)
# --- END GlobalAveragePool --- #

def GlobalLpPool(info: Dict[str, Any]) -> LayerAndActivationBase:
    print(f"{stack()[0].function()} args received: {info}")
    raise NotImplementedError("Not implemented")
# --- END GlobalLpPool --- #

def GlobalMaxPool(info: Dict[str, Any]) -> LayerAndActivationBase:
    print(f"{stack()[0].function()} args received: {info}")
    raise NotImplementedError("Not implemented")
# --- END GlobalMaxPool --- #

def Greater(info: Dict[str, Any]) -> LayerAndActivationBase:
    print(f"{stack()[0].function()} args received: {info}")
    raise NotImplementedError("Not implemented")
# --- END Greater --- #

def GreaterOrEqual(info: Dict[str, Any]) -> LayerAndActivationBase:
    print(f"{stack()[0].function()} args received: {info}")
    raise NotImplementedError("Not implemented")
# --- END GreaterOrEqual --- #

def GridSample(info: Dict[str, Any]) -> LayerAndActivationBase:
    print(f"{stack()[0].function()} args received: {info}")
    raise NotImplementedError("Not implemented")
# --- END GridSample --- #

def GroupNormalization(info: Dict[str, Any]) -> LayerAndActivationBase:
    print(f"{stack()[0].function()} args received: {info}")
    raise NotImplementedError("Not implemented")
# --- END GroupNormalization --- #
