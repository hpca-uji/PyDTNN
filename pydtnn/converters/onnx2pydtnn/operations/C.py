# Typing related (or non important) imports
from typing import *
from pydtnn.layers.layer_and_activation_base import LayerAndActivationBase
from inspect import stack # This is only in order to get the function's name

# Functionality imports
import pydtnn.layers as layer
import pydtnn.converters.onnx2pydtnn.constants as cons

def Cast(info: Dict[str, Any]) -> LayerAndActivationBase:
    print(f"Operation: {stack()[0].function}\nargs received: {info}")
    raise NotImplementedError("Not implemented")
# --- END Cast --- #

def CastLike(info: Dict[str, Any]) -> LayerAndActivationBase:
    print(f"Operation: {stack()[0].function}\nargs received: {info}")
    raise NotImplementedError("Not implemented")
# --- END CastLike --- #

def Ceil(info: Dict[str, Any]) -> LayerAndActivationBase:
    print(f"Operation: {stack()[0].function}\nargs received: {info}")
    raise NotImplementedError("Not implemented")
# --- END Ceil --- #

def Celu(info: Dict[str, Any]) -> LayerAndActivationBase:
    print(f"Operation: {stack()[0].function}\nargs received: {info}")
    raise NotImplementedError("Not implemented")
# --- END Celu --- #

def CenterCropPad(info: Dict[str, Any]) -> LayerAndActivationBase:
    print(f"Operation: {stack()[0].function}\nargs received: {info}")
    raise NotImplementedError("Not implemented")
# --- END CenterCropPad --- #

def Clip(info: Dict[str, Any]) -> LayerAndActivationBase:
    print(f"Operation: {stack()[0].function}\nargs received: {info}")
    raise NotImplementedError("Not implemented")
# --- END Clip --- #

def Col2Im(info: Dict[str, Any]) -> LayerAndActivationBase:
    print(f"Operation: {stack()[0].function}\nargs received: {info}")
    raise NotImplementedError("Not implemented")
# --- END Col2Im --- #

def Compress(info: Dict[str, Any]) -> LayerAndActivationBase:
    print(f"Operation: {stack()[0].function}\nargs received: {info}")
    raise NotImplementedError("Not implemented")
# --- END Compress --- #

def Concat(info: Dict[str, Any]) -> LayerAndActivationBase:
    print(f"Operation: {stack()[0].function}")
    print(f"attributes: {info[cons.CONST_ATTRIBUTES]}")
    # Onnx attributes names from: https://onnx.ai/onnx/operators/onnx__Concat.html#l-onnx-doc-concat
    ONNX_AXIS = "axis"
    # There are no PyDTNN attributes names from ConcatenationBlock class.
    
    # TODO: Check if this class is correct
    list_concat_nodes = info[cons.CONST_LISTS_NODES]

    return layer.ConcatenationBlock(list_concat_nodes)   
# --- END Concat --- #

def ConcatFromSequence(info: Dict[str, Any]) -> LayerAndActivationBase:
    print(f"Operation: {stack()[0].function}\nargs received: {info}")
    raise NotImplementedError("Not implemented")
# --- END ConcatFromSequence --- #

def Constant(info: Dict[str, Any]) -> LayerAndActivationBase:
    print(f"Operation: {stack()[0].function}\nargs received: {info}")
    raise NotImplementedError("Not implemented")
# --- END Constant --- #

def ConstantOfShape(info: Dict[str, Any]) -> LayerAndActivationBase:
    print(f"Operation: {stack()[0].function}\nargs received: {info}")
    raise NotImplementedError("Not implemented")
# --- END ConstantOfShape --- #

def Conv(info: Dict[str, Any]) -> LayerAndActivationBase:
    
    print(f"Operation: {stack()[0].function}")
    print(f"attributes: {info[cons.CONST_ATTRIBUTES]}")

    # Onnx attributes names from: https://onnx.ai/onnx/operators/onnx__Conv.html#l-onnx-doc-conv
    ONNX_COUNT_DILATATIONS = "dilations"
    ONNX_GROUP = "group" 
    ONNX_KERNEL_SHAPE = "kernel_shape"
    ONNX_PADS = "pads"
    ONNX_STRIDES = "strides"
    # PyDTNN attributes names from Conv2D class.
    PYDTNN_DILATION = "dilation"
    PYDTNN_NFILTERS = "nfilters"
    PYDTNN_FILTER_SHAPE = "filter_shape"
    PYDTNN_PADDING = "padding"
    PYDTNN_STRIDE = "stride"    
    
    args = dict()
    dict_attributes = info[cons.CONST_ATTRIBUTES]

    if ONNX_COUNT_DILATATIONS in dict_attributes:
        args[PYDTNN_DILATION] = dict_attributes[ONNX_COUNT_DILATATIONS]
    if ONNX_GROUP in dict_attributes:
        # TODO: Check if this is correct:
        args[PYDTNN_NFILTERS] = dict_attributes[ONNX_GROUP]
    if ONNX_KERNEL_SHAPE in dict_attributes:
        args[PYDTNN_FILTER_SHAPE] = dict_attributes[ONNX_KERNEL_SHAPE]
    if ONNX_PADS in dict_attributes:
        args[PYDTNN_PADDING] = cons.pads_from_onnx_to_pydtnn(pads = dict_attributes[ONNX_PADS])
    if ONNX_STRIDES in dict_attributes:
        args[PYDTNN_STRIDE] = dict_attributes[ONNX_STRIDES]
    
    # TODO: Look if it's necessary to set the Bias here.

    # TODO: Borrar
    print("CONVOLUCION")
    for k in args.keys():
        print(f"args[{k}]: {type(args[k])} | {args[k]}")

    return layer.Conv2D(**args)
# --- END Conv --- #

def ConvInteger(info: Dict[str, Any]) -> LayerAndActivationBase:
    print(f"Operation: {stack()[0].function}\nargs received: {info}")
    raise NotImplementedError("Not implemented")
# --- END ConvInteger --- #

def ConvTranspose(info: Dict[str, Any]) -> LayerAndActivationBase:
    print(f"Operation: {stack()[0].function}\nargs received: {info}")
    raise NotImplementedError("Not implemented")
# --- END ConvTranspose --- #

def Cos(info: Dict[str, Any]) -> LayerAndActivationBase:
    print(f"Operation: {stack()[0].function}\nargs received: {info}")
    raise NotImplementedError("Not implemented")
# --- END Cos --- #

def Cosh(info: Dict[str, Any]) -> LayerAndActivationBase:
    print(f"Operation: {stack()[0].function}\nargs received: {info}")
    raise NotImplementedError("Not implemented")
# --- END Cosh --- #

def CumSum(info: Dict[str, Any]) -> LayerAndActivationBase:
    print(f"Operation: {stack()[0].function}\nargs received: {info}")
    raise NotImplementedError("Not implemented")
# --- END CumSum --- #
