# Typing related (or non important) imports
from typing import *
from pydtnn.layers.layer_and_activation_base import LayerAndActivationBase
from inspect import stack # This is only in order to get the function's name

# Functionality imports
import pydtnn.layers

def Abs(info: Dict[str, Any]) -> List[LayerAndActivationBase]:
    print(f"{stack()[0].function()} args received: {info}")
    raise NotImplementedError("Not implemented")
# --- END Abs --- #

def Acos(info: Dict[str, Any]) -> List[LayerAndActivationBase]:
    print(f"{stack()[0].function()} args received: {info}")
    raise NotImplementedError("Not implemented")
# --- END Acos --- #

def Acosh(info: Dict[str, Any]) -> List[LayerAndActivationBase]:
    print(f"{stack()[0].function()} args received: {info}")
    raise NotImplementedError("Not implemented")
# --- END Acosh --- #

def Add(info: Dict[str, Any]) -> List[LayerAndActivationBase]:
    print(f"{stack()[0].function()} args received: {info}")
    raise NotImplementedError("Not implemented")
    #AdditionBlock
# --- END Add --- #

def AffineGrid(info: Dict[str, Any]) -> List[LayerAndActivationBase]:
    print(f"{stack()[0].function()} args received: {info}")
    raise NotImplementedError("Not implemented")
# --- END AffineGrid --- #

def And(info: Dict[str, Any]) -> List[LayerAndActivationBase]:
    print(f"{stack()[0].function()} args received: {info}")
    raise NotImplementedError("Not implemented")
# --- END And --- #

def ArgMax(info: Dict[str, Any]) -> List[LayerAndActivationBase]:
    print(f"{stack()[0].function()} args received: {info}")
    raise NotImplementedError("Not implemented")
# --- END ArgMax --- #

def ArgMin(info: Dict[str, Any]) -> List[LayerAndActivationBase]:
    print(f"{stack()[0].function()} args received: {info}")
    raise NotImplementedError("Not implemented")
# --- END ArgMin --- #

def Asin(info: Dict[str, Any]) -> List[LayerAndActivationBase]:
    print(f"{stack()[0].function()} args received: {info}")
    raise NotImplementedError("Not implemented")
# --- END Asin --- #

def Asinh(info: Dict[str, Any]) -> List[LayerAndActivationBase]:
    print(f"{stack()[0].function()} args received: {info}")
    raise NotImplementedError("Not implemented")
# --- END Asinh --- #

def Atan(info: Dict[str, Any]) -> List[LayerAndActivationBase]:
    print(f"{stack()[0].function()} args received: {info}")
    raise NotImplementedError("Not implemented")
# --- END Atan --- #

def Atanh(info: Dict[str, Any]) -> List[LayerAndActivationBase]:
    print(f"{stack()[0].function()} args received: {info}")
    raise NotImplementedError("Not implemented")
# --- END Atanh --- #

def AveragePool(info: Dict[str, Any]) -> List[LayerAndActivationBase]:
    print(f"{stack()[0].function()} args received: {info}")
    raise NotImplementedError("Not implemented")
    #AveragePool2D
# --- END AveragePool --- #
