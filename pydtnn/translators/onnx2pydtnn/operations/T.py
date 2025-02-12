# Typing related (or non important) imports
from typing import *
from pydtnn.layers.layer_and_activation_base import LayerAndActivationBase
from inspect import stack # This is only in order to get the function's name

# Functionality imports
# EMPTY (for now)

def Tan(info: Dict[str, Any]) -> LayerAndActivationBase:
    print(f"{stack()[0].function} args received: {info}")
    raise NotImplementedError("Not implemented")
# --- END Tan --- #

def Tanh(info: Dict[str, Any]) -> LayerAndActivationBase:
    print(f"{stack()[0].function} args received: {info}")
    raise NotImplementedError("Not implemented")
# --- END Tanh --- #

def TfIdfVectorizer(info: Dict[str, Any]) -> LayerAndActivationBase:
    print(f"{stack()[0].function} args received: {info}")
    raise NotImplementedError("Not implemented")
# --- END TfIdfVectorizer --- #

def ThresholdedRelu(info: Dict[str, Any]) -> LayerAndActivationBase:
    print(f"{stack()[0].function} args received: {info}")
    raise NotImplementedError("Not implemented")
# --- END ThresholdedRelu --- #

def Tile(info: Dict[str, Any]) -> LayerAndActivationBase:
    print(f"{stack()[0].function} args received: {info}")
    raise NotImplementedError("Not implemented")
# --- END Tile --- #

def TopK(info: Dict[str, Any]) -> LayerAndActivationBase:
    print(f"{stack()[0].function} args received: {info}")
    raise NotImplementedError("Not implemented")
# --- END TopK --- #

def Transpose(info: Dict[str, Any]) -> LayerAndActivationBase:
    print(f"{stack()[0].function} args received: {info}")
    raise NotImplementedError("Not implemented")
# --- END Transpose --- #

def Trilu(info: Dict[str, Any]) -> LayerAndActivationBase:
    print(f"{stack()[0].function} args received: {info}")
    raise NotImplementedError("Not implemented")
# --- END Trilu --- #
