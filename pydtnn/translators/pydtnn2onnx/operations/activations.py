# Typing-related imports
from typing import *
from onnx import NodeProto
from inspect import stack # This is only in order to get the function's name

# Funtionality imports
# Empty (for now)

def Arctanh(info:dict[str, Any])-> NodeProto:
    print(f"Operation: {stack()[0].function}\nargs received: {info}")
    raise NotImplementedError("Not implemented")
# --- END Arctanh --- #
    
def Log(info:dict[str, Any])-> NodeProto:
    print(f"Operation: {stack()[0].function}\nargs received: {info}")
    raise NotImplementedError("Not implemented")
# --- END Log --- #
    
def Relu(info:dict[str, Any])-> NodeProto:
    print(f"Operation: {stack()[0].function}\nargs received: {info}")
    raise NotImplementedError("Not implemented")
# --- END Relu --- #
    
def Sigmoid(info:dict[str, Any])-> NodeProto:
    print(f"Operation: {stack()[0].function}\nargs received: {info}")
    raise NotImplementedError("Not implemented")
# --- END Sigmoid --- #
    
def Softmax(info:dict[str, Any])-> NodeProto:
    print(f"Operation: {stack()[0].function}\nargs received: {info}")
    raise NotImplementedError("Not implemented")
# --- END Softmax --- #
    
def Tanh(info:dict[str, Any])-> NodeProto:
    print(f"Operation: {stack()[0].function}\nargs received: {info}")
    raise NotImplementedError("Not implemented")
# --- END Tanh --- #
    