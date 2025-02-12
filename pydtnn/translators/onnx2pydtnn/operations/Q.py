# Typing related (or non important) imports
from typing import *
from pydtnn.layers.layer_and_activation_base import LayerAndActivationBase
from inspect import stack # This is only in order to get the function's name

# Functionality imports
# EMPTY (for now)

def QLinearConv(info: Dict[str, Any]) -> LayerAndActivationBase:
    print(f"{stack()[0].function} args received: {info}")
    raise NotImplementedError("Not implemented")
# --- END QLinearConv --- #

def QLinearMatMul(info: Dict[str, Any]) -> LayerAndActivationBase:
    print(f"{stack()[0].function} args received: {info}")
    raise NotImplementedError("Not implemented")
# --- END QLinearMatMul --- #

def QuantizeLinear(info: Dict[str, Any]) -> LayerAndActivationBase:
    print(f"{stack()[0].function} args received: {info}")
    raise NotImplementedError("Not implemented")
# --- END QuantizeLinear --- #
