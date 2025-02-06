# Typing related (or non important) imports
from typing import *
from pydtnn.layers.layer_and_activation_base import LayerAndActivationBase
from inspect import stack # This is only in order to get the function's name

# Functionality imports
# EMPTY (for now)

def MatMul(info: Dict[str, Any]) -> List[LayerAndActivationBase]:
    print(f"{stack()[0].function()} args received: {info}")
    raise NotImplementedError("Not implemented")
# --- END MatMul --- #

def MatMulInteger(info: Dict[str, Any]) -> List[LayerAndActivationBase]:
    print(f"{stack()[0].function()} args received: {info}")
    raise NotImplementedError("Not implemented")
# --- END MatMulInteger --- #

def Max(info: Dict[str, Any]) -> List[LayerAndActivationBase]:
    print(f"{stack()[0].function()} args received: {info}")
    raise NotImplementedError("Not implemented")
# --- END Max --- #

def MaxPool(info: Dict[str, Any]) -> List[LayerAndActivationBase]:
    print(f"{stack()[0].function()} args received: {info}")
    raise NotImplementedError("Not implemented")
    # MaxPool2D
# --- END MaxPool --- #

def MaxRoiPool(info: Dict[str, Any]) -> List[LayerAndActivationBase]:
    print(f"{stack()[0].function()} args received: {info}")
    raise NotImplementedError("Not implemented")
# --- END MaxRoiPool --- #

def MaxUnpool(info: Dict[str, Any]) -> List[LayerAndActivationBase]:
    print(f"{stack()[0].function()} args received: {info}")
    raise NotImplementedError("Not implemented")
# --- END MaxUnpool --- #

def Mean(info: Dict[str, Any]) -> List[LayerAndActivationBase]:
    print(f"{stack()[0].function()} args received: {info}")
    raise NotImplementedError("Not implemented")
# --- END Mean --- #

def MeanVarianceNormalization(info: Dict[str, Any]) -> List[LayerAndActivationBase]:
    print(f"{stack()[0].function()} args received: {info}")
    raise NotImplementedError("Not implemented")
# --- END MeanVarianceNormalization --- #

def MelWeightMatrix(info: Dict[str, Any]) -> List[LayerAndActivationBase]:
    print(f"{stack()[0].function()} args received: {info}")
    raise NotImplementedError("Not implemented")
# --- END MelWeightMatrix --- #

def Min(info: Dict[str, Any]) -> List[LayerAndActivationBase]:
    print(f"{stack()[0].function()} args received: {info}")
    raise NotImplementedError("Not implemented")
# --- END Min --- #

def Mish(info: Dict[str, Any]) -> List[LayerAndActivationBase]:
    print(f"{stack()[0].function()} args received: {info}")
    raise NotImplementedError("Not implemented")
# --- END Mish --- #

def Mod(info: Dict[str, Any]) -> List[LayerAndActivationBase]:
    print(f"{stack()[0].function()} args received: {info}")
    raise NotImplementedError("Not implemented")
# --- END Mod --- #

def Mul(info: Dict[str, Any]) -> List[LayerAndActivationBase]:
    print(f"{stack()[0].function()} args received: {info}")
    raise NotImplementedError("Not implemented")
# --- END Mul --- #

def Multinomial(info: Dict[str, Any]) -> List[LayerAndActivationBase]:
    print(f"{stack()[0].function()} args received: {info}")
    raise NotImplementedError("Not implemented")
# --- END Multinomial --- #
