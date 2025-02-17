from typing import *

# PyDTNN imports:
from pydtnn.activations import *
from pydtnn.layers import *
from operations import activations as act

CONST_INPUTS = "inputs"
CONST_NUM_OP = "num_operation"
CONST_WEIGHTS = "weights"
CONST_LAYER = "layer"
CONST_OP_NAME = "op_name"

# TODO: para cada clase hay que:    
#  -> obtener las entradas (y los tipos de estas)
#  -> Poner un nombre a las salidas
#  -> Si es una que tenga ".paths", hay que extraer todas las capas y ponerlas a continuación

def make_output_name(op_name: str, num_operations: int) -> str:
    return "_".join([op_name, str(num_operations)])
# --- END make_output_name --- #

def pads_from_onnx_to_pydttn(pads: List[int]) -> Tuple[int, int]: #-> List[Tuple[int, int]]:
        # "pads format should be as follow [x1_begin, x2_begin…x1_end, x2_end,…]" from, for example, https://onnx.ai/onnx/operators/onnx__AveragePool.html
        # Onnx: [x1_begin, x2_begin, ..., x1_end, x2_end, ...] ==> "PyDTNN: [(x1_begin, x1_end), (x2_end, x2_begin), ...]"
        # ==> PyDTNN only admits a int or a (vpadding, hpadding) ==> It's assumed that is the first tuple.

        num_pads = len(pads)//2
        _pads = [(0,0)] * (num_pads)
        for i in range(num_pads):
            _pads[i] = (pads[i], pads[i + num_pads])

        return _pads
# --- END pads_from_onnx_to_pydttn --- #

def get_layer_name_and_id(name: str) -> Tuple[str, int]:
    TYPE_DEVICE = ["GPU", "CPU"]
    ID_NAME_SEPARATOR = "_"

    index_separator = name.index(ID_NAME_SEPARATOR)

    layer_name = name[index_separator + 1:]
    id = int(name[:index_separator])
    for type in TYPE_DEVICE:
        layer_name = layer_name.replace(type, "")

    return layer_name, id
# --- END get_layer_name --- #

def not_implemented():
    raise NotImplemented("Por hacer")
# --- END not_implemented --- #

SWITCH_OPERATION_PYDTNN_TO_ONNX = {
    # activations:
    "Arctanh": act.make_activation_node,
    "Log": act.make_activation_node,
    "Relu": act.make_activation_node,
    "Sigmoid": act.make_activation_node,
    "Softmax": act.Softmax,
    "Tanh": act.make_activation_node,

    # layers:
    "AdditionBlock": not_implemented(),
    "AveragePool2D": not_implemented(),
    "BatchNormalizationRelu": not_implemented(),
    "BatchNormalization": not_implemented(),
    "ConcatenationBlock": not_implemented(),
    "Conv2DBatchNormalizationRelu": not_implemented(),
    "Conv2DBatchNormalization": not_implemented(),
    "Conv2DRelu": not_implemented(),
    "Conv2D": not_implemented(),
    "Dropout": not_implemented(),
    "FC": not_implemented(),
    "Flatten": not_implemented(),
    "Input": not_implemented(),
    "MaxPool2D": not_implemented(),

}