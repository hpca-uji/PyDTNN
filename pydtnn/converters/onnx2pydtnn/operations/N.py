# Typing related (or non important) imports
from typing import *
from pydtnn.layers.layer_and_activation_base import LayerAndActivationBase
from inspect import stack # This is only in order to get the function's name

# Functionality imports
# EMPTY (for now)

def Neg(info: Dict[str, Any]) -> LayerAndActivationBase:
    print(f"Operation: {stack()[0].function}\nargs received: {info}")
    raise NotImplementedError("Not implemented")
# --- END Neg --- #

def NegativeLogLikelihoodLoss(info: Dict[str, Any]) -> LayerAndActivationBase:
    print(f"Operation: {stack()[0].function}\nargs received: {info}")
    raise NotImplementedError("Not implemented")
# --- END NegativeLogLikelihoodLoss --- #

def NonMaxSuppression(info: Dict[str, Any]) -> LayerAndActivationBase:
    print(f"Operation: {stack()[0].function}\nargs received: {info}")
    raise NotImplementedError("Not implemented")
# --- END NonMaxSuppression --- #

def NonZero(info: Dict[str, Any]) -> LayerAndActivationBase:
    print(f"Operation: {stack()[0].function}\nargs received: {info}")
    raise NotImplementedError("Not implemented")
# --- END NonZero --- #

def Not(info: Dict[str, Any]) -> LayerAndActivationBase:
    print(f"Operation: {stack()[0].function}\nargs received: {info}")
    raise NotImplementedError("Not implemented")
# --- END Not --- #
