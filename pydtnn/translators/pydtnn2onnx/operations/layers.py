# Typing-related imports
from typing import *
from inspect import stack # This is only in order to get the function's name

# Funtionality imports
from onnx import NodeProto
from onnx.helper import make_node
import constants as cons

def AdditionBlock(info: dict[str, Any]) -> NodeProto:
    print(f"Operation: {stack()[0].function}\nargs received: {info}")
    
    # TODO: Esta capa tiene un atributo "paths" que contiene una lista de adds.
    info[cons.CONST_LAYER] = pass

    raise NotImplementedError("Not implemented")
# --- END AdditionBlock --- #

def AveragePool2D(info: dict[str, Any]) -> NodeProto:
    print(f"Operation: {stack()[0].function}\nargs received: {info}")
    
    op_name = info[cons.CONST_OP_NAME]
    num_operation = info[cons.CONST_NUM_OPERATION]
    inputs = info[cons.CONST_INPUTS]
    outputs = cons.make_output_name(op_name, num_operation)

    attribute = {"dilations": 1, 
                 "kernel_shape": 1,
                 "pads": 1,
                 "strides": 1,
                 }

    return make_node(op_type="Softmax", 
                     inputs=inputs, outputs=outputs, name=outputs, 
                     doc_string="Read https://onnx.ai/onnx/operators/onnx__Softmax.html and https://github.com/hpca-uji/PyDTNN for more information.",
                     kwargs=attribute)
# --- END AveragePool2D --- #

def BatchNormalizationRelu(info: dict[str, Any]) -> NodeProto:
    print(f"Operation: {stack()[0].function}\nargs received: {info}")
    raise NotImplementedError("Not implemented")
# --- END BatchNormalizationRelu --- #

def BatchNormalization(info: dict[str, Any]) -> NodeProto:
    print(f"Operation: {stack()[0].function}\nargs received: {info}")
    raise NotImplementedError("Not implemented")
# --- END BatchNormalization --- #

def ConcatenationBlock(info: dict[str, Any]) -> NodeProto:
    print(f"Operation: {stack()[0].function}\nargs received: {info}")
    raise NotImplementedError("Not implemented")
# --- END ConcatenationBlock --- #

def Conv2DBatchNormalizationRelu(info: dict[str, Any]) -> NodeProto:
    print(f"Operation: {stack()[0].function}\nargs received: {info}")
    raise NotImplementedError("Not implemented")
# --- END Conv2DBatchNormalizationRelu --- #

def Conv2DBatchNormalization(info: dict[str, Any]) -> NodeProto:
    print(f"Operation: {stack()[0].function}\nargs received: {info}")
    raise NotImplementedError("Not implemented")
# --- END Conv2DBatchNormalization --- #

def Conv2DRelu(info: dict[str, Any]) -> NodeProto:
    print(f"Operation: {stack()[0].function}\nargs received: {info}")
    raise NotImplementedError("Not implemented")
# --- END Conv2DRelu --- #

def Conv2D(info: dict[str, Any]) -> NodeProto:
    print(f"Operation: {stack()[0].function}\nargs received: {info}")
    raise NotImplementedError("Not implemented")
# --- END Conv2D --- #

def Dropout(info: dict[str, Any]) -> NodeProto:
    print(f"Operation: {stack()[0].function}\nargs received: {info}")
    raise NotImplementedError("Not implemented")
# --- END Dropout --- #

def FC(info: dict[str, Any]) -> NodeProto:
    print(f"Operation: {stack()[0].function}\nargs received: {info}")
    raise NotImplementedError("Not implemented")
# --- END FC --- #

def Flatten(info: dict[str, Any]) -> NodeProto:
    print(f"Operation: {stack()[0].function}\nargs received: {info}")
    raise NotImplementedError("Not implemented")
# --- END Flatten --- #

def Input(info: dict[str, Any]) -> NodeProto:
    print(f"Operation: {stack()[0].function}\nargs received: {info}")
    raise NotImplementedError("Not implemented")
# --- END Input --- #

def MaxPool2D(info: dict[str, Any]) -> NodeProto:
    print(f"Operation: {stack()[0].function}\nargs received: {info}")
    raise NotImplementedError("Not implemented")
# --- END MaxPool2D --- #
