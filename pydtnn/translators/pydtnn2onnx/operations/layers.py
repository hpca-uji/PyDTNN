# Typing-related imports
from typing import *
from onnx import NodeProto
from inspect import stack # This is only in order to get the function's name

# Funtionality imports
# Empty (for now)

def AdditionBlock(info: dict[str, Any]) -> NodeProto:
    print(f"Operation: {stack()[0].function}\nargs received: {info}")
    raise NotImplementedError("Not implemented")
# --- END AdditionBlock --- #

def AveragePool2D(info: dict[str, Any]) -> NodeProto:
    print(f"Operation: {stack()[0].function}\nargs received: {info}")
    raise NotImplementedError("Not implemented")
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
