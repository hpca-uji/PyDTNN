# Typing related (or non important) imports
from typing import *
from pydtnn.layers.layer_and_activation_base import LayerAndActivationBase
from inspect import stack # This is only in order to get the function's name

# Functionality imports
import pydtnn.translators.onnx2pydtnn.constants as cons
import pydtnn.layers as layer

def Unique(info: Dict[str, Any]) -> LayerAndActivationBase:
    print(f"Operation: {stack()[0].function}\nargs received: {info}")
    raise NotImplementedError("Not implemented")
# --- END Unique --- #

def Unsqueeze(info: Dict[str, Any]) -> LayerAndActivationBase:
    # Onnx information: https://onnx.ai/onnx/operators/onnx__Unsqueeze.html
    print(f"Operation: {stack()[0].function}")
    print(f"attributes: {info[cons.CONST_ATTRIBUTES]}")
    ONNX_AXES = "axes"

    PYDTNN_AXES = "axis"
    dict_attributes = info[cons.CONST_ATTRIBUTES]
    
    args = {}

    if ONNX_AXES in dict_attributes:
        args[PYDTNN_AXES] = dict_attributes[ONNX_AXES]

    # TODO: Move it to a file and do it in the right way.

    from pydtnn.tracers import PYDTNN_OPS_EVENT, PYDTNN_OPS_EVENTS, PYDTNN_MDL_FORWARD, PYDTNN_MDL_BACKWARD, \
    PYDTNN_MDL_EVENT, PYDTNN_MDL_EVENTS, PYDTNN_OPS_BACKWARD_ELTW_SUM, PYDTNN_OPS_FORWARD_ELTW_SUM
    from numpy import expand_dims
    from pydtnn.layers import Layer
    from pydtnn.utils import encode_tensor

    class _Unsqueeze(Layer):

        def __init__(self, shape=(1,), axis=()):
            super().__init__(shape)
            self.axis = axis
        # - END __init__ - #

        def initialize(self, prev_shape, need_dx=False):
            super().initialize(prev_shape, need_dx)
            self.shape = encode_tensor(self.shape, self.model.tensor_format)

        def initialize_block_layer(self):
            super().initialize_block_layer()
            assert all([o == self.out_shapes[0] for o in self.out_shapes])
            self.shape = self.out_shapes[0]
        # - END initialize_block_layer - #

        def forward(self, x):
            return expand_dims(x, axis=self.axis)
    # -- END _Unsqueeze -- #

    return _Unsqueeze(**args)
    
# --- END Unsqueeze --- #

def Upsample(info: Dict[str, Any]) -> LayerAndActivationBase:
    print(f"Operation: {stack()[0].function}\nargs received: {info}")
    raise NotImplementedError("Not implemented")
# --- END Upsample --- #
