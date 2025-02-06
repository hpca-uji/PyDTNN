# Typing related (or non important) imports
from typing import *
from pydtnn.layers.layer_and_activation_base import LayerAndActivationBase
from inspect import stack # This is only in order to get the function's name

# Functionality imports
# EMPTY (for now)

def Adagrad(info: Dict[str, Any]) -> List[LayerAndActivationBase]:
    print(f"{stack()[0].function()} args received: {info}")
    raise NotImplementedError("Not implemented")
# --- END Adagrad --- #

def Adam(info: Dict[str, Any]) -> List[LayerAndActivationBase]:
    print(f"{stack()[0].function()} args received: {info}")
    raise NotImplementedError("Not implemented")
# --- END Adam --- #

def Gradient(info: Dict[str, Any]) -> List[LayerAndActivationBase]:
    print(f"{stack()[0].function()} args received: {info}")
    raise NotImplementedError("Not implemented")
# --- END Gradient --- #

def Momentum(info: Dict[str, Any]) -> List[LayerAndActivationBase]:
    print(f"{stack()[0].function()} args received: {info}")
    raise NotImplementedError("Not implemented")
# --- END Momentum --- #
