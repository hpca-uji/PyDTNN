# Typing related (or non important) imports
from typing import *
from pydtnn.layers.layer_and_activation_base import LayerAndActivationBase
from inspect import stack # This is only in order to get the function's name

# Functionality imports
# EMPTY (for now)

def Cast(info: Dict[str, Any]) -> List[LayerAndActivationBase]:
    print(f"{stack()[0].function()} args received: {info}")
    raise NotImplementedError("Not implemented")
# --- END Cast --- #

def CastLike(info: Dict[str, Any]) -> List[LayerAndActivationBase]:
    print(f"{stack()[0].function()} args received: {info}")
    raise NotImplementedError("Not implemented")
# --- END CastLike --- #

def Ceil(info: Dict[str, Any]) -> List[LayerAndActivationBase]:
    print(f"{stack()[0].function()} args received: {info}")
    raise NotImplementedError("Not implemented")
# --- END Ceil --- #

def Celu(info: Dict[str, Any]) -> List[LayerAndActivationBase]:
    print(f"{stack()[0].function()} args received: {info}")
    raise NotImplementedError("Not implemented")
# --- END Celu --- #

def CenterCropPad(info: Dict[str, Any]) -> List[LayerAndActivationBase]:
    print(f"{stack()[0].function()} args received: {info}")
    raise NotImplementedError("Not implemented")
# --- END CenterCropPad --- #

def Clip(info: Dict[str, Any]) -> List[LayerAndActivationBase]:
    print(f"{stack()[0].function()} args received: {info}")
    raise NotImplementedError("Not implemented")
# --- END Clip --- #

def Col2Im(info: Dict[str, Any]) -> List[LayerAndActivationBase]:
    print(f"{stack()[0].function()} args received: {info}")
    raise NotImplementedError("Not implemented")
# --- END Col2Im --- #

def Compress(info: Dict[str, Any]) -> List[LayerAndActivationBase]:
    print(f"{stack()[0].function()} args received: {info}")
    raise NotImplementedError("Not implemented")
# --- END Compress --- #

def Concat(info: Dict[str, Any]) -> List[LayerAndActivationBase]:
    print(f"{stack()[0].function()} args received: {info}")
    raise NotImplementedError("Not implemented")
    # ConcatenationBlock
# --- END Concat --- #

def ConcatFromSequence(info: Dict[str, Any]) -> List[LayerAndActivationBase]:
    print(f"{stack()[0].function()} args received: {info}")
    raise NotImplementedError("Not implemented")
# --- END ConcatFromSequence --- #

def Constant(info: Dict[str, Any]) -> List[LayerAndActivationBase]:
    print(f"{stack()[0].function()} args received: {info}")
    raise NotImplementedError("Not implemented")
# --- END Constant --- #

def ConstantOfShape(info: Dict[str, Any]) -> List[LayerAndActivationBase]:
    print(f"{stack()[0].function()} args received: {info}")
    raise NotImplementedError("Not implemented")
# --- END ConstantOfShape --- #

def Conv(info: Dict[str, Any]) -> List[LayerAndActivationBase]:
    print(f"{stack()[0].function()} args received: {info}")
    raise NotImplementedError("Not implemented")
    #Conv2D
# --- END Conv --- #

def ConvInteger(info: Dict[str, Any]) -> List[LayerAndActivationBase]:
    print(f"{stack()[0].function()} args received: {info}")
    raise NotImplementedError("Not implemented")
# --- END ConvInteger --- #

def ConvTranspose(info: Dict[str, Any]) -> List[LayerAndActivationBase]:
    print(f"{stack()[0].function()} args received: {info}")
    raise NotImplementedError("Not implemented")
# --- END ConvTranspose --- #

def Cos(info: Dict[str, Any]) -> List[LayerAndActivationBase]:
    print(f"{stack()[0].function()} args received: {info}")
    raise NotImplementedError("Not implemented")
# --- END Cos --- #

def Cosh(info: Dict[str, Any]) -> List[LayerAndActivationBase]:
    print(f"{stack()[0].function()} args received: {info}")
    raise NotImplementedError("Not implemented")
# --- END Cosh --- #

def CumSum(info: Dict[str, Any]) -> List[LayerAndActivationBase]:
    print(f"{stack()[0].function()} args received: {info}")
    raise NotImplementedError("Not implemented")
# --- END CumSum --- #
