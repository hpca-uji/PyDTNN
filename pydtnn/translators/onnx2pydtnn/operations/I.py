# Typing related (or non important) imports
from typing import *
from pydtnn.layers.layer_and_activation_base import LayerAndActivationBase
from inspect import stack # This is only in order to get the function's name

# Functionality imports
# EMPTY (for now)

def Identity(info: Dict[str, Any]) -> LayerAndActivationBase:
    print(f"Operation: {stack()[0].function}\nargs received: {info}")
    raise NotImplementedError("Not implemented")
# --- END Identity --- #

def If(info: Dict[str, Any]) -> LayerAndActivationBase:
    print(f"Operation: {stack()[0].function}\nargs received: {info}")
    raise NotImplementedError("Not implemented")
# --- END If --- #

def ImageDecoder(info: Dict[str, Any]) -> LayerAndActivationBase:
    print(f"Operation: {stack()[0].function}\nargs received: {info}")
    raise NotImplementedError("Not implemented")
# --- END ImageDecoder --- #

def InstanceNormalization(info: Dict[str, Any]) -> LayerAndActivationBase:
    print(f"Operation: {stack()[0].function}\nargs received: {info}")
    raise NotImplementedError("Not implemented")
# --- END InstanceNormalization --- #

def IsInf(info: Dict[str, Any]) -> LayerAndActivationBase:
    print(f"Operation: {stack()[0].function}\nargs received: {info}")
    raise NotImplementedError("Not implemented")
# --- END IsInf --- #

def IsNaN(info: Dict[str, Any]) -> LayerAndActivationBase:
    print(f"Operation: {stack()[0].function}\nargs received: {info}")
    raise NotImplementedError("Not implemented")
# --- END IsNaN --- #
