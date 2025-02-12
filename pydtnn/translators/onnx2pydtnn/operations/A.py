# Typing related (or non important) imports
from typing import *
from pydtnn.layers.layer_and_activation_base import LayerAndActivationBase
from inspect import stack # This is only in order to get the function's name

# Functionality imports
import pydtnn.layers as layer
import pydtnn.translators.onnx2pydtnn.constants as cons

def Abs(info: Dict[str, Any]) -> LayerAndActivationBase:
    print(f"{stack()[0].function()} args received: {info}")
    raise NotImplementedError("Not implemented")
# --- END Abs --- #

def Acos(info: Dict[str, Any]) -> LayerAndActivationBase:
    print(f"{stack()[0].function()} args received: {info}")
    raise NotImplementedError("Not implemented")
# --- END Acos --- #

def Acosh(info: Dict[str, Any]) -> LayerAndActivationBase:
    print(f"{stack()[0].function()} args received: {info}")
    raise NotImplementedError("Not implemented")
# --- END Acosh --- #

def Add(info: Dict[str, Any]) -> LayerAndActivationBase:

    # TODO: from print to "log - debug" or somthing like that.
    print(f"{stack()[0].function()} args received: {info}")
    list_adding_nodes = info[cons.CONST_LISTS_NODES]

    return layer.AdditionBlock(list_adding_nodes)
# --- END Add --- #

def AffineGrid(info: Dict[str, Any]) -> LayerAndActivationBase:
    print(f"{stack()[0].function()} args received: {info}")
    raise NotImplementedError("Not implemented")
# --- END AffineGrid --- #

def And(info: Dict[str, Any]) -> LayerAndActivationBase:
    print(f"{stack()[0].function()} args received: {info}")
    raise NotImplementedError("Not implemented")
# --- END And --- #

def ArgMax(info: Dict[str, Any]) -> LayerAndActivationBase:
    print(f"{stack()[0].function()} args received: {info}")
    raise NotImplementedError("Not implemented")
# --- END ArgMax --- #

def ArgMin(info: Dict[str, Any]) -> LayerAndActivationBase:
    print(f"{stack()[0].function()} args received: {info}")
    raise NotImplementedError("Not implemented")
# --- END ArgMin --- #

def Asin(info: Dict[str, Any]) -> LayerAndActivationBase:
    print(f"{stack()[0].function()} args received: {info}")
    raise NotImplementedError("Not implemented")
# --- END Asin --- #

def Asinh(info: Dict[str, Any]) -> LayerAndActivationBase:
    print(f"{stack()[0].function()} args received: {info}")
    raise NotImplementedError("Not implemented")
# --- END Asinh --- #

def Atan(info: Dict[str, Any]) -> LayerAndActivationBase:
    print(f"{stack()[0].function()} args received: {info}")
    raise NotImplementedError("Not implemented")
# --- END Atan --- #

def Atanh(info: Dict[str, Any]) -> LayerAndActivationBase:
    print(f"{stack()[0].function()} args received: {info}")
    raise NotImplementedError("Not implemented")
# --- END Atanh --- #

def AveragePool(info: Dict[str, Any]) -> LayerAndActivationBase:

    # Onnx attributes names from: https://onnx.ai/onnx/operators/onnx__AveragePool.html
    ONNX_COUNT_DILATATIONS = "dilations"
    ONNX_KERNEL_SHAPE = "kernel_shape"
    ONNX_PADS = "pads"
    ONNX_STRIDES = "strides"
    # PyDTNN attributes names from AbstractPool2DLayer class.
    PYDTNN_DILATION = "dilation"
    PYDTNN_POOL_SHAPE = "pool_shape"
    PYDTNN_PADDING = "padding"
    PYDTNN_STRIDE = "stride"    

    
    print(f"{stack()[0].function()} args received: {info}")
    
    dict_attributes = info[cons.CONST_ATTRIBUTES]
    args = dict()

    if ONNX_COUNT_DILATATIONS in dict_attributes:
        args[PYDTNN_POOL_SHAPE] = dict_attributes[ONNX_KERNEL_SHAPE]
    if ONNX_KERNEL_SHAPE in dict_attributes:
        args[PYDTNN_PADDING] = dict_attributes[ONNX_PADS]
    if ONNX_PADS in dict_attributes:
        args[PYDTNN_STRIDE] = dict_attributes[ONNX_STRIDES]
    if ONNX_STRIDES in dict_attributes:
        args[PYDTNN_DILATION] = dict_attributes[ONNX_COUNT_DILATATIONS]

    return layer.AveragePool2D(*args)
# --- END AveragePool --- #
