# Typing related (or non important) imports
from typing import *
from pydtnn.layers.layer_and_activation_base import LayerAndActivationBase
from inspect import stack # This is only in order to get the function's name

# Functionality imports
import pydtnn.layers

def BatchNormalization(info: Dict[str, Any]) -> List[LayerAndActivationBase]:
    print(f"{stack()[0].function()} args received: {info}")
    raise NotImplementedError("Not implemented")
    #BatchNormalization
# --- END BatchNormalization --- #

def Bernoulli(info: Dict[str, Any]) -> List[LayerAndActivationBase]:
    print(f"{stack()[0].function()} args received: {info}")
    raise NotImplementedError("Not implemented")
# --- END Bernoulli --- #

def BitShift(info: Dict[str, Any]) -> List[LayerAndActivationBase]:
    print(f"{stack()[0].function()} args received: {info}")
    raise NotImplementedError("Not implemented")
# --- END BitShift --- #

def BitwiseAnd(info: Dict[str, Any]) -> List[LayerAndActivationBase]:
    print(f"{stack()[0].function()} args received: {info}")
    raise NotImplementedError("Not implemented")
# --- END BitwiseAnd --- #

def BitwiseNot(info: Dict[str, Any]) -> List[LayerAndActivationBase]:
    print(f"{stack()[0].function()} args received: {info}")
    raise NotImplementedError("Not implemented")
# --- END BitwiseNot --- #

def BitwiseOr(info: Dict[str, Any]) -> List[LayerAndActivationBase]:
    print(f"{stack()[0].function()} args received: {info}")
    raise NotImplementedError("Not implemented")
# --- END BitwiseOr --- #

def BitwiseXor(info: Dict[str, Any]) -> List[LayerAndActivationBase]:
    print(f"{stack()[0].function()} args received: {info}")
    raise NotImplementedError("Not implemented")
# --- END BitwiseXor --- #

def BlackmanWindow(info: Dict[str, Any]) -> List[LayerAndActivationBase]:
    print(f"{stack()[0].function()} args received: {info}")
    raise NotImplementedError("Not implemented")
# --- END BlackmanWindow --- #
