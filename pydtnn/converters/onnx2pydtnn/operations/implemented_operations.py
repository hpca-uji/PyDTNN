# Typing related (or non important) imports
from typing import Any
from pydtnn.layer_base import LayerBase

# Functionality imports
import pydtnn.layers as layer
import pydtnn.activations as activation
import pydtnn.converters.onnx2pydtnn.constants as cons

# ========================= #

# ===== #
# = A = #
# ===== #


def Add(info: dict[str, Any]) -> LayerBase:

    # TODO: from print to "log - debug" or somthing like that.
    print(f"attributes: {info[cons.CONST_ATTRIBUTES]}")
    list_adding_nodes = info[cons.CONST_listS_NODES]

    from pydtnn.layers.addition_block import AdditionBlock
    return AdditionBlock(list_adding_nodes)
# --- END Add --- #


def AveragePool(info: dict[str, Any]) -> LayerBase:

    # Onnx attributes names from: https://onnx.ai/onnx/operators/onnx__AveragePool.html
    ONNX_COUNT_DILATATIONS = "dilations"
    ONNX_KERNEL_SHAPE = "kernel_shape"
    ONNX_PADS = "pads"
    ONNX_STRIDES = "strides"
    # PyDTNN attributes names from AbstractPool2DLayer class.
    PYDTNN_DILATION = "dilation"
    PYDTNN_POOL_SHAPE = "pool_shape"
    PYDTNN_PADDING = "padding"
    PYDTNN_STRIDE = "stride"

    print(f"attributes: {info[cons.CONST_ATTRIBUTES]}")

    dict_attributes = info[cons.CONST_ATTRIBUTES]
    args = dict()

    if ONNX_COUNT_DILATATIONS in dict_attributes:
        args[PYDTNN_DILATION] = dict_attributes[ONNX_COUNT_DILATATIONS]
    if ONNX_KERNEL_SHAPE in dict_attributes:
        args[PYDTNN_POOL_SHAPE] = dict_attributes[ONNX_KERNEL_SHAPE]
    if ONNX_PADS in dict_attributes:
        args[PYDTNN_PADDING] = cons.pads_from_onnx_to_pydtnn(pads=dict_attributes[ONNX_PADS])
    if ONNX_STRIDES in dict_attributes:
        args[PYDTNN_STRIDE] = dict_attributes[ONNX_STRIDES]

    from pydtnn.layers.average_pool_2d import AveragePool2D
    return AveragePool2D(**args)
# --- END AveragePool --- #

# ========================= #


# ===== #
# = B = #
# ===== #

def BatchNormalization(info: dict[str, Any]) -> LayerBase:
    print(f"attributes: {info[cons.CONST_ATTRIBUTES]}")

    # Onnx attributes names from: https://onnx.ai/onnx/operators/onnx__BatchNormalization.html#l-onnx-doc-batchnormalization
    ONNX_EPSILON = "epsilon"
    ONNX_MOMENTUM = "momentum"
    # PyDTNN attributes names from BatchNormalization class.
    PYDTNN_EPSILON = "epsilon"
    PYDTNN_MOMENTUM = "momentum"

    args = dict()
    dict_attributes = info[cons.CONST_ATTRIBUTES]

    if ONNX_EPSILON in dict_attributes:
        args[PYDTNN_EPSILON] = dict_attributes[ONNX_EPSILON]
    if ONNX_MOMENTUM in dict_attributes:
        args[PYDTNN_MOMENTUM] = dict_attributes[ONNX_MOMENTUM]

    from pydtnn.layers.batch_normalization import BatchNormalization
    return BatchNormalization(**args)
# --- END BatchNormalization --- #

# ========================= #


# ===== #
# = C = #
# ===== #

def Concat(info: dict[str, Any]) -> LayerBase:
    print(f"attributes: {info[cons.CONST_ATTRIBUTES]}")
    # Onnx attributes names from: https://onnx.ai/onnx/operators/onnx__Concat.html#l-onnx-doc-concat
    ONNX_AXIS = "axis"
    # There are no PyDTNN attributes names from ConcatenationBlock class.

    # TODO: Check if this class is correct
    list_concat_nodes = info[cons.CONST_listS_NODES]

    from pydtnn.layers.concatenation_block import ConcatenationBlock
    return ConcatenationBlock(list_concat_nodes)
# --- END Concat --- #


def Conv(info: dict[str, Any]) -> LayerBase:

    print(f"attributes: {info[cons.CONST_ATTRIBUTES]}")

    # Onnx attributes names from: https://onnx.ai/onnx/operators/onnx__Conv.html#l-onnx-doc-conv
    ONNX_COUNT_DILATATIONS = "dilations"
    ONNX_GROUP = "group"
    ONNX_KERNEL_SHAPE = "kernel_shape"
    ONNX_PADS = "pads"
    ONNX_STRIDES = "strides"
    # PyDTNN attributes names from Conv2D class.
    PYDTNN_DILATION = "dilation"
    PYDTNN_NFILTERS = "nfilters"
    PYDTNN_FILTER_SHAPE = "filter_shape"
    PYDTNN_PADDING = "padding"
    PYDTNN_STRIDE = "stride"

    args = dict()
    dict_attributes = info[cons.CONST_ATTRIBUTES]

    if ONNX_COUNT_DILATATIONS in dict_attributes:
        args[PYDTNN_DILATION] = dict_attributes[ONNX_COUNT_DILATATIONS]
    if ONNX_GROUP in dict_attributes:
        # TODO: Check if this is correct:
        args[PYDTNN_NFILTERS] = dict_attributes[ONNX_GROUP]
    if ONNX_KERNEL_SHAPE in dict_attributes:
        args[PYDTNN_FILTER_SHAPE] = dict_attributes[ONNX_KERNEL_SHAPE]
    if ONNX_PADS in dict_attributes:
        args[PYDTNN_PADDING] = cons.pads_from_onnx_to_pydtnn(pads=dict_attributes[ONNX_PADS])
    if ONNX_STRIDES in dict_attributes:
        args[PYDTNN_STRIDE] = dict_attributes[ONNX_STRIDES]

    # TODO: Look if it's necessary to set the Bias here.

    # TODO: Borrar
    print("CONVOLUCION")
    for k in args.keys():
        print(f"args[{k}]: {type(args[k])} | {args[k]}")

    from pydtnn.layers.conv_2d import Conv2D
    return Conv2D(**args)
# --- END Conv --- #

# ========================= #


# ===== #
# = D = #
# ===== #

def Dropout(info: dict[str, Any]) -> LayerBase:
    print(f"attributes: {info[cons.CONST_ATTRIBUTES]}")
    # Onnx attributes names from: https://onnx.ai/onnx/operators/onnx__Dropout.html#l-onnx-doc-dropout
    ONNX_SEED = "seed"  # TODO: Check if the random seed it's important. If it is, check how to set it.
    # PyDTNN attributes names from Dropout class.
    PYDTNN_RATE = "rate"

    args = {}

    # TODO: Check if this is correct.
    # Droput can receive 3 inputs: the previous layer output [Tensor],
    #   the ratio (of random dropout) [Float] and if it's in training mode [bool]
    # Then if it has more than one input and it's not a bool or the previous layer output, it is the ratio.
    _other_inputs = set(enumerate(info[cons.CONST_ALL_INPUTS])) - set(enumerate(info[cons.CONST_INPUTS]))
    other_inputs = [elem[1] for elem in sorted(_other_inputs, key=lambda x: x[0])]

    if len(other_inputs) > 0:
        for k in other_inputs:
            elem = info[cons.CONST_WEIGHTS][other_inputs[k]]
            if not isinstance(elem, bool):
                args[PYDTNN_RATE] = elem
                break

    print(f"==> args: {args}")

    from pydtnn.layers.dropout import Dropout
    return Dropout(**args)
# --- END Dropout --- #

# ========================= #

# ===== #
# = F = #
# ===== #


def Flatten(info: dict[str, Any]) -> LayerBase:
    print(f"attributes: {info[cons.CONST_ATTRIBUTES]}")
    # Source: https://onnx.ai/onnx/operators/onnx__Flatten.html
    # It has one attribute (axis), but there is no equivalence in PyDTNN.
    # ==> In PyDTNN the axis is always 1.
    from pydtnn.layers.flatten import Flatten
    return Flatten()
# --- END Flatten --- #

# ========================= #

# ===== #
# = G = #
# ===== #


def Gemm(info: dict[str, Any]) -> LayerBase:

    print(f"attributes: {info[cons.CONST_ATTRIBUTES]}")
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

    dict_attributes = info[cons.CONST_ATTRIBUTES]

    alpha = dict_attributes[ONNX_ALPHA] if ONNX_ALPHA in dict_attributes else 1.0
    beta = dict_attributes[ONNX_BETA] if ONNX_BETA in dict_attributes else 1.0
    transA = dict_attributes[ONNX_TRANS_A] if ONNX_TRANS_A in dict_attributes else None
    transB = dict_attributes[ONNX_TRANS_B] if ONNX_TRANS_B in dict_attributes else None

    # TODO: make this programming terrorism into an actual class or classes
    from pydtnn.layers.fc import FC
    pseudo_gemm = FC()

    _other_inputs = set(enumerate(info[cons.CONST_ALL_INPUTS])) - set(enumerate(info[cons.CONST_INPUTS]))
    other_inputs = [elem[1] for elem in sorted(_other_inputs, key=lambda x: x[0])]

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

    pseudo_gemm.weights_initializer = _weights_initializer  # (lambda *x: b.T if transB is not None else b)
    if c is not None:
        pseudo_gemm.biases_initializer = _biases_initializer
    pseudo_gemm.forward = _mod_forward

    return pseudo_gemm
# --- END Gemm --- #


def GlobalAveragePool(info: dict[str, Any]) -> LayerBase:
    print(f"attributes: {info[cons.CONST_ATTRIBUTES]}")
    # 1.- Onnx documentation: https://onnx.ai/onnx/operators/onnx__GlobalAveragePool.html

    # PyDTNN attributes names from AbstractPool2DLayer class.
    PYDTNN_POOL_SHAPE = "pool_shape"
    PYDTNN_STRIDE = "stride"

    args = dict()

    operations = info[cons.CONST_PREV_LAYERS]
    _input = info[cons.CONST_INPUTS][0]  # It should be a list with only one input

    # TODO: check if this is correct.

    # "This is equivalent to AveragePool with kernel size equal to the spatial dimension of input tensor." [1]
    args[PYDTNN_POOL_SHAPE] = operations[_input].shape
    args[PYDTNN_STRIDE] = 1

    from pydtnn.layers.average_pool_2d import AveragePool2D
    return AveragePool2D(**args)
# --- END GlobalAveragePool --- #


# ========================= #


# ===== #
# = M = #
# ===== #

def MaxPool(info: dict[str, Any]) -> LayerBase:
    print("------")
    print(f"attributes: {info[cons.CONST_ATTRIBUTES]}")

    # Onnx attributes names from: https://onnx.ai/onnx/operators/onnx__MaxPool.html#l-onnx-doc-maxpool
    ONNX_KERNEL_SHAPE = "kernel_shape"
    ONNX_PADS = "pads"
    ONNX_STRIDES = "strides"
    ONNX_COUNT_DILATATIONS = "dilations"
    # PyDTNN attributes names from AbstractPool2DLayer class.
    PYDTNN_POOL_SHAPE = "pool_shape"
    PYDTNN_PADDING = "padding"
    PYDTNN_STRIDE = "stride"
    PYDTNN_DILATION = "dilation"

    print("DEBUG")  # TODO: ¡¡BORRAR!!

    dict_attributes = info[cons.CONST_ATTRIBUTES]
    args = dict()

    if ONNX_COUNT_DILATATIONS in dict_attributes:
        args[PYDTNN_DILATION] = dict_attributes[ONNX_COUNT_DILATATIONS]
    if ONNX_KERNEL_SHAPE in dict_attributes:
        args[PYDTNN_POOL_SHAPE] = dict_attributes[ONNX_KERNEL_SHAPE]
    if ONNX_PADS in dict_attributes:
        args[PYDTNN_PADDING] = cons.pads_from_onnx_to_pydtnn(pads=dict_attributes[ONNX_PADS])
    if ONNX_STRIDES in dict_attributes:
        args[PYDTNN_STRIDE] = dict_attributes[ONNX_STRIDES]

    for k in args.keys():
        print(f"args[{k}]: {type(args[k])} | {args[k]}")
        a = args[k]

    from pydtnn.layers.max_pool_2d import MaxPool2D
    return MaxPool2D(**args)
# --- END MaxPool --- #


def Mul(info: dict[str, Any]) -> LayerBase:
    print(f"attributes: {info[cons.CONST_ATTRIBUTES]}")

    # TODO: Move it to a file and do it in the right way.

    from numpy import multiply
    from pydtnn.layers.abstract.block_layer import AbstractBlockLayer

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
                    x[i] = layer.forward(x[i])

                if i > 0:
                    # TODO: do it with Cython.
                    x[0] = multiply(x[0], x[i])
            return x[0]
        # - END forward - #

        def backward(self, dy):
            dx = [dy] * len(self.paths)
            for i, p in enumerate(self.paths):
                for layer in reversed(p):
                    dx[i] = layer.backward(dx[i])
                if i > 0:
                    # TODO: do it with Cython adn chekc if it's correct.
                    dx[0] = multiply(dx[0], dx[i])
            return dx[0]
        # - END backward - #
    # -- END _Mul -- #

    return _Mul(info[cons.CONST_listS_NODES])
# --- END Mul --- #

# ========================= #


# ===== #
# = R = #
# ===== #

def Relu(info: dict[str, Any]) -> LayerBase:
    # ONNX info: https://onnx.ai/onnx/operators/onnx__Relu.html
    print(f"attributes: {info[cons.CONST_ATTRIBUTES]}")
    from pydtnn.activations.relu import Relu
    return Relu()
# --- END Relu --- #


# ========================= #


# ===== #
# = U = #
# ===== #


def Unsqueeze(info: dict[str, Any]) -> LayerBase:
    # Onnx information: https://onnx.ai/onnx/operators/onnx__Unsqueeze.html
    print(f"attributes: {info[cons.CONST_ATTRIBUTES]}")
    ONNX_AXES = "axes"

    PYDTNN_AXES = "axis"
    dict_attributes = info[cons.CONST_ATTRIBUTES]

    args = {}

    if ONNX_AXES in dict_attributes:
        args[PYDTNN_AXES] = dict_attributes[ONNX_AXES]

    # TODO: Move it to a file and do it in the right way.

    from numpy import expand_dims
    from pydtnn.layers.layer import Layer

    class _Unsqueeze(Layer):

        def __init__(self, shape=(1,), axis=()):
            super().__init__(shape)
            self.axis = axis
        # - END __init__ - #

        def initialize(self, prev_shape, need_dx=False):
            super().initialize(prev_shape, need_dx)
            self.shape = self.shape + self.model.encode_shape(self.model.tensor_format)

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
