# Typing related (or non important) imports
from typing import *
from pydtnn.layers.layer_and_activation_base import LayerAndActivationBase
from inspect import stack # This is only in order to get the function's name

# Functionality imports
# EMPTY (for now)

def OneHot(info: Dict[str, Any]) -> LayerAndActivationBase:
    print(f"{stack()[0].function} args received: {info}")
    raise NotImplementedError("Not implemented")
# --- END OneHot --- #

def _Optional(info: Dict[str, Any]) -> LayerAndActivationBase:
    print(f"{stack()[0].function} args received: {info}")
    raise NotImplementedError("Not implemented")
# --- END Optional --- #

def OptionalGetElement(info: Dict[str, Any]) -> LayerAndActivationBase:
    print(f"{stack()[0].function} args received: {info}")
    raise NotImplementedError("Not implemented")
# --- END OptionalGetElement --- #

def OptionalHasElement(info: Dict[str, Any]) -> LayerAndActivationBase:
    print(f"{stack()[0].function} args received: {info}")
    raise NotImplementedError("Not implemented")
# --- END OptionalHasElement --- #

def Or(info: Dict[str, Any]) -> LayerAndActivationBase:
    print(f"{stack()[0].function} args received: {info}")
    raise NotImplementedError("Not implemented")
# --- Or --- #END
