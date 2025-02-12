# Typing related (or non important) imports
from typing import *
from pydtnn.layers.layer_and_activation_base import LayerAndActivationBase
from inspect import stack # This is only in order to get the function's name

# Functionality imports
import pydtnn.layers as layer
import pydtnn.translators.onnx2pydtnn.constants as cons

def GRU(info: Dict[str, Any]) -> LayerAndActivationBase:
    print(f"{stack()[0].function()} args received: {info}")
    raise NotImplementedError("Not implemented")
# --- END GRU --- #

def Gather(info: Dict[str, Any]) -> LayerAndActivationBase:
    print(f"{stack()[0].function()} args received: {info}")
    raise NotImplementedError("Not implemented")
# --- END Gather --- #

def GatherElements(info: Dict[str, Any]) -> LayerAndActivationBase:
    print(f"{stack()[0].function()} args received: {info}")
    raise NotImplementedError("Not implemented")
# --- END GatherElements --- #

def GatherND(info: Dict[str, Any]) -> LayerAndActivationBase:
    print(f"{stack()[0].function()} args received: {info}")
    raise NotImplementedError("Not implemented")
# --- END GatherND --- #

def Gelu(info: Dict[str, Any]) -> LayerAndActivationBase:
    print(f"{stack()[0].function()} args received: {info}")
    raise NotImplementedError("Not implemented")
# --- END Gelu --- #

def Gemm(info: Dict[str, Any]) -> LayerAndActivationBase:

    print(f"{stack()[0].function()} args received: {info}")
    # Onnx documentation: https://onnx.ai/onnx/operators/onnx__Gemm.html
    ONNX_ALPHA = "alpha"
    ONNX_BETA = "beta"
    ONNX_TRANS_A = "transA"
    ONNX_TRANS_B = "transB"

    # FC' PyDTNN0' implementation:
    #   res = self.model.matmul(x, self.weights)
    #   self.model.tracer.emit_event(PYDTNN_OPS_EVENT, 0)
    #   return res + self.biases if self.use_bias else 0

    # ONNX Gemm implementation:
    #   A’ = transpose(A) if transA else A
    #   B’ = transpose(B) if transB else B
    #   Y = alpha * A’ * B’ + beta * C
    # B: PyDTNN's weights
    # C: PyDTNN's bias    

    attributes = info[cons.CONST_ATTRIBUTES]

    alpha = attributes[ONNX_ALPHA] if ONNX_ALPHA in attributes else 1.0
    beta = attributes[ONNX_BETA] if ONNX_BETA in attributes else 1.0
    transA = attributes[ONNX_TRANS_A] if ONNX_TRANS_A in attributes else None
    transB = attributes[ONNX_TRANS_B] if ONNX_TRANS_B in attributes else None

    # TODO: make this programming terrorism into an actual class or classes
    pseudo_gemm = layer.FC()
    
    other_inputs = set(info[cons.CONST_WEIGHTS].keys()) - set(info[cons.CONST_INPUTS])

    if len(other_inputs) == 1:
        b = info[cons.CONST_WEIGHTS][other_inputs[0]]
        c = None
    else: 
        b = info[cons.CONST_WEIGHTS][other_inputs[0]]
        c = info[cons.CONST_WEIGHTS][other_inputs[1]]

    original_fw = pseudo_gemm.forward

    def _weights_initializer(*to_ignore):
        return b.T if transB is not None else b

    def _biases_initializer(*to_ignore):
        return beta * c

    def _mod_forward(x):
        x = alpha * (x.T if transA is not None else x)
        original_fw(x)

    pseudo_gemm.weights_initializer = _weights_initializer # (lambda *x: b.T if transB is not None else b) 
    if c is not None:
        pseudo_gemm.biases_initializer = _biases_initializer 
    pseudo_gemm.forward = _mod_forward

    return pseudo_gemm
# --- END Gemm --- #

def GlobalAveragePool(info: Dict[str, Any]) -> LayerAndActivationBase:
    print(f"{stack()[0].function()} args received: {info}")
    # 1.- Onnx documentation: https://onnx.ai/onnx/operators/onnx__GlobalAveragePool.html
    
    # PyDTNN attributes names from AbstractPool2DLayer class.
    PYDTNN_POOL_SHAPE = "pool_shape"
    PYDTNN_STRIDE = "stride"

    print(f"{stack()[0].function()} args received: {info}")
    args = dict()

    operations = info[cons.CONST_PREV_LAYERS]
    _input = info[cons.CONST_INPUTS][0] # It should be a list with only one input

    # TODO: check if this is correct.

    # "This is equivalent to AveragePool with kernel size equal to the spatial dimension of input tensor." [1]
    args[PYDTNN_POOL_SHAPE] = operations[_input].shape
    args[PYDTNN_STRIDE] = 1
    
    return layer.AveragePool2D(*args)
# --- END GlobalAveragePool --- #

def GlobalLpPool(info: Dict[str, Any]) -> LayerAndActivationBase:
    print(f"{stack()[0].function()} args received: {info}")
    raise NotImplementedError("Not implemented")
# --- END GlobalLpPool --- #

def GlobalMaxPool(info: Dict[str, Any]) -> LayerAndActivationBase:
    print(f"{stack()[0].function()} args received: {info}")
    raise NotImplementedError("Not implemented")
# --- END GlobalMaxPool --- #

def Greater(info: Dict[str, Any]) -> LayerAndActivationBase:
    print(f"{stack()[0].function()} args received: {info}")
    raise NotImplementedError("Not implemented")
# --- END Greater --- #

def GreaterOrEqual(info: Dict[str, Any]) -> LayerAndActivationBase:
    print(f"{stack()[0].function()} args received: {info}")
    raise NotImplementedError("Not implemented")
# --- END GreaterOrEqual --- #

def GridSample(info: Dict[str, Any]) -> LayerAndActivationBase:
    print(f"{stack()[0].function()} args received: {info}")
    raise NotImplementedError("Not implemented")
# --- END GridSample --- #

def GroupNormalization(info: Dict[str, Any]) -> LayerAndActivationBase:
    print(f"{stack()[0].function()} args received: {info}")
    raise NotImplementedError("Not implemented")
# --- END GroupNormalization --- #
