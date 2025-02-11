# Typing related (or non important) imports
from typing import *
from pydtnn.layers.layer_and_activation_base import LayerAndActivationBase
from inspect import stack # This is only in order to get the function's name

# Functionality imports
import pydtnn.layers as layer
from constants import CONST_INPUTS, CONST_ATTRIBUTES, CONST_PREV_LAYERS

def MatMul(info: Dict[str, Any]) -> LayerAndActivationBase:
    print(f"{stack()[0].function()} args received: {info}")
    raise NotImplementedError("Not implemented")
# --- END MatMul --- #

def MatMulInteger(info: Dict[str, Any]) -> LayerAndActivationBase:
    print(f"{stack()[0].function()} args received: {info}")
    raise NotImplementedError("Not implemented")
# --- END MatMulInteger --- #

def Max(info: Dict[str, Any]) -> LayerAndActivationBase:
    print(f"{stack()[0].function()} args received: {info}")
    raise NotImplementedError("Not implemented")
# --- END Max --- #

def MaxPool(info: Dict[str, Any]) -> LayerAndActivationBase:
    print(f"{stack()[0].function()} args received: {info}")
    
    # Onnx attributes names from: https://onnx.ai/onnx/operators/onnx__MaxPool.html#l-onnx-doc-maxpool
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
    
    dict_attributes = info[CONST_ATTRIBUTES]
    args = dict()

    if ONNX_COUNT_DILATATIONS in dict_attributes:
        args[PYDTNN_POOL_SHAPE] = dict_attributes[ONNX_KERNEL_SHAPE]
    if ONNX_KERNEL_SHAPE in dict_attributes:
        args[PYDTNN_PADDING] = dict_attributes[ONNX_PADS]
    if ONNX_PADS in dict_attributes:
        args[PYDTNN_STRIDE] = dict_attributes[ONNX_STRIDES]
    if ONNX_STRIDES in dict_attributes:
        args[PYDTNN_DILATION] = dict_attributes[ONNX_COUNT_DILATATIONS]

    return layer.MaxPool2D(*args)
# --- END MaxPool --- #

def MaxRoiPool(info: Dict[str, Any]) -> LayerAndActivationBase:
    print(f"{stack()[0].function()} args received: {info}")
    raise NotImplementedError("Not implemented")
# --- END MaxRoiPool --- #

def MaxUnpool(info: Dict[str, Any]) -> LayerAndActivationBase:
    print(f"{stack()[0].function()} args received: {info}")
    raise NotImplementedError("Not implemented")
# --- END MaxUnpool --- #

def Mean(info: Dict[str, Any]) -> LayerAndActivationBase:
    print(f"{stack()[0].function()} args received: {info}")
    raise NotImplementedError("Not implemented")
# --- END Mean --- #

def MeanVarianceNormalization(info: Dict[str, Any]) -> LayerAndActivationBase:
    print(f"{stack()[0].function()} args received: {info}")
    raise NotImplementedError("Not implemented")
# --- END MeanVarianceNormalization --- #

def MelWeightMatrix(info: Dict[str, Any]) -> LayerAndActivationBase:
    print(f"{stack()[0].function()} args received: {info}")
    raise NotImplementedError("Not implemented")
# --- END MelWeightMatrix --- #

def Min(info: Dict[str, Any]) -> LayerAndActivationBase:
    print(f"{stack()[0].function()} args received: {info}")
    raise NotImplementedError("Not implemented")
# --- END Min --- #

def Mish(info: Dict[str, Any]) -> LayerAndActivationBase:
    print(f"{stack()[0].function()} args received: {info}")
    raise NotImplementedError("Not implemented")
# --- END Mish --- #

def Mod(info: Dict[str, Any]) -> LayerAndActivationBase:
    print(f"{stack()[0].function()} args received: {info}")
    raise NotImplementedError("Not implemented")
# --- END Mod --- #

def Mul(info: Dict[str, Any]) -> LayerAndActivationBase:
    print(f"{stack()[0].function()} args received: {info}")
    # TODO: Let's see how to do it.
    raise NotImplementedError("Not implemented")
# --- END Mul --- #

def Multinomial(info: Dict[str, Any]) -> LayerAndActivationBase:
    print(f"{stack()[0].function()} args received: {info}")
    raise NotImplementedError("Not implemented")
# --- END Multinomial --- #
