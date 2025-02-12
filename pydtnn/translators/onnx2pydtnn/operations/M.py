# Typing related (or non important) imports
from typing import *
from pydtnn.layers.layer_and_activation_base import LayerAndActivationBase
from inspect import stack # This is only in order to get the function's name

# Functionality imports
import pydtnn.layers as layer
import pydtnn.translators.onnx2pydtnn.constants as cons

def MatMul(info: Dict[str, Any]) -> LayerAndActivationBase:
    print(f"Operation: {stack()[0].function}\nargs received: {info}")
    raise NotImplementedError("Not implemented")
# --- END MatMul --- #

def MatMulInteger(info: Dict[str, Any]) -> LayerAndActivationBase:
    print(f"Operation: {stack()[0].function}\nargs received: {info}")
    raise NotImplementedError("Not implemented")
# --- END MatMulInteger --- #

def Max(info: Dict[str, Any]) -> LayerAndActivationBase:
    print(f"Operation: {stack()[0].function}\nargs received: {info}")
    raise NotImplementedError("Not implemented")
# --- END Max --- #

def MaxPool(info: Dict[str, Any]) -> LayerAndActivationBase:
    print(f"Operation: {stack()[0].function}\nargs received: {info}")
    
    # Onnx attributes names from: https://onnx.ai/onnx/operators/onnx__MaxPool.html#l-onnx-doc-maxpool    
    ONNX_KERNEL_SHAPE = "kernel_shape"
    ONNX_PADS = "pads"
    ONNX_STRIDES = "strides"
    # PyDTNN attributes names from AbstractPool2DLayer class.    
    PYDTNN_POOL_SHAPE = "pool_shape"
    PYDTNN_PADDING = "padding"
    PYDTNN_STRIDE = "stride"    
    PYDTNN_DILATION = "dilation"
    ONNX_COUNT_DILATATIONS = "dilations"

    print(f"Operation: {stack()[0].function}\nargs received: {info}")
    
    dict_attributes = info[cons.CONST_ATTRIBUTES]
    args = dict()

    if ONNX_COUNT_DILATATIONS in dict_attributes: 
        args[PYDTNN_DILATION] = dict_attributes[ONNX_COUNT_DILATATIONS]        
    if ONNX_KERNEL_SHAPE in dict_attributes: 
        args[PYDTNN_POOL_SHAPE] = dict_attributes[ONNX_KERNEL_SHAPE]        
    if ONNX_PADS in dict_attributes: 
        print(f"dict_attributes[ONNX_PADS]: {dict_attributes[ONNX_PADS]}")
        args[PYDTNN_PADDING] = cons.pads_from_onnx_to_pydttn(pads = dict_attributes[ONNX_PADS])
        print(f"args[PYDTNN_PADDING]: {args[PYDTNN_PADDING]}")
    if ONNX_STRIDES in dict_attributes: 
        args[PYDTNN_STRIDE] = dict_attributes[ONNX_STRIDES]
    
    return layer.MaxPool2D(**args)
# --- END MaxPool --- #

def MaxRoiPool(info: Dict[str, Any]) -> LayerAndActivationBase:
    print(f"Operation: {stack()[0].function}\nargs received: {info}")
    raise NotImplementedError("Not implemented")
# --- END MaxRoiPool --- #

def MaxUnpool(info: Dict[str, Any]) -> LayerAndActivationBase:
    print(f"Operation: {stack()[0].function}\nargs received: {info}")
    raise NotImplementedError("Not implemented")
# --- END MaxUnpool --- #

def Mean(info: Dict[str, Any]) -> LayerAndActivationBase:
    print(f"Operation: {stack()[0].function}\nargs received: {info}")
    raise NotImplementedError("Not implemented")
# --- END Mean --- #

def MeanVarianceNormalization(info: Dict[str, Any]) -> LayerAndActivationBase:
    print(f"Operation: {stack()[0].function}\nargs received: {info}")
    raise NotImplementedError("Not implemented")
# --- END MeanVarianceNormalization --- #

def MelWeightMatrix(info: Dict[str, Any]) -> LayerAndActivationBase:
    print(f"Operation: {stack()[0].function}\nargs received: {info}")
    raise NotImplementedError("Not implemented")
# --- END MelWeightMatrix --- #

def Min(info: Dict[str, Any]) -> LayerAndActivationBase:
    print(f"Operation: {stack()[0].function}\nargs received: {info}")
    raise NotImplementedError("Not implemented")
# --- END Min --- #

def Mish(info: Dict[str, Any]) -> LayerAndActivationBase:
    print(f"Operation: {stack()[0].function}\nargs received: {info}")
    raise NotImplementedError("Not implemented")
# --- END Mish --- #

def Mod(info: Dict[str, Any]) -> LayerAndActivationBase:
    print(f"Operation: {stack()[0].function}\nargs received: {info}")
    raise NotImplementedError("Not implemented")
# --- END Mod --- #

def Mul(info: Dict[str, Any]) -> LayerAndActivationBase:
    print(f"Operation: {stack()[0].function}\nargs received: {info}")

    # TODO: Move it to a file and do it in the right way.

    from pydtnn.tracers import PYDTNN_OPS_EVENT, PYDTNN_OPS_EVENTS, PYDTNN_MDL_FORWARD, PYDTNN_MDL_BACKWARD, \
    PYDTNN_MDL_EVENT, PYDTNN_MDL_EVENTS, PYDTNN_OPS_BACKWARD_ELTW_SUM, PYDTNN_OPS_FORWARD_ELTW_SUM
    from numpy import multiply
    from pydtnn.layers.abstract_block_layer import AbstractBlockLayer

    class _Mul(AbstractBlockLayer):
        def initialize_block_layer(self):
            super().initialize_block_layer()
            assert all([o == self.out_shapes[0] for o in self.out_shapes])
            self.shape = self.out_shapes[0]
        # - END initialize_block_layer - #

        def forward(self, x):
            x = [x] * len(self.paths)
            for i, p in enumerate(self.paths):
                for layer in p:
                    self.model.tracer.emit_event(PYDTNN_MDL_EVENT, layer.id * PYDTNN_MDL_EVENTS + PYDTNN_MDL_FORWARD)
                    x[i] = layer.forward(x[i])
                    self.model.tracer.emit_event(PYDTNN_MDL_EVENT, 0)
                
                if i > 0:
                    self.model.tracer.emit_event(PYDTNN_OPS_EVENT,
                                             self.id * PYDTNN_OPS_EVENTS + PYDTNN_OPS_FORWARD_ELTW_SUM)
                    # TODO: do it with Cython.
                    x[0] = multiply(x[0], x[i])
                    self.model.tracer.emit_event(PYDTNN_OPS_EVENT, 0)
            return x[0]
        # - END forward - #

        def backward(self, dy):
            dx = [dy] * len(self.paths)
            for i, p in enumerate(self.paths):
                for layer in reversed(p):
                    self.model.tracer.emit_event(PYDTNN_MDL_EVENT, layer.id * PYDTNN_MDL_EVENTS + PYDTNN_MDL_BACKWARD)
                    dx[i] = layer.backward(dx[i])
                    self.model.tracer.emit_event(PYDTNN_MDL_EVENT, 0)
                if i > 0:
                    self.model.tracer.emit_event(PYDTNN_OPS_EVENT,
                                                    self.id * PYDTNN_OPS_EVENTS + PYDTNN_OPS_BACKWARD_ELTW_SUM)
                    # TODO: do it with Cython adn chekc if it's correct.
                    dx[0] = multiply(dx[0], dx[i])
                    self.model.tracer.emit_event(PYDTNN_OPS_EVENT, 0)
            return dx[0]
        # - END backward - #
    # -- END _Mul -- #

    return _Mul(info[cons.CONST_LISTS_NODES])
# --- END Mul --- #

def Multinomial(info: Dict[str, Any]) -> LayerAndActivationBase:
    print(f"Operation: {stack()[0].function}\nargs received: {info}")
    raise NotImplementedError("Not implemented")
# --- END Multinomial --- #
