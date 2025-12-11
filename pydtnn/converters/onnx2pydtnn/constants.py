# ONNX operations:
from .operations.implemented_operations import *

from typing import Callable
from pydtnn.layer_base import LayerBase

CONST_NODE = "node"
CONST_OPSET = "opset_version"
CONST_OUPTUS = "outputs"
CONST_ATTRIBUTES = "attributes"
CONST_INPUTS = "inputs"
CONST_ALL_INPUTS = "all_inputs"
CONST_listS_NODES = "lists_nodes"
CONST_WEIGHTS = "weights"
CONST_PREV_LAYERS = "previous_layers"

# Operations to do:
# DenseNet169 - {'Conv', 'BatchNormalization', 'Unsqueeze', 'Add', 'Mul', 'Relu', 'MaxPool', 'AveragePool', 'GlobalAveragePool', 'Concat'}
# ResNet50 - {'Conv', 'MaxPool', 'Relu', 'Add', 'BatchNormalization', 'GlobalAveragePool', 'Gemm', 'Flatten'}
# VGG19 - {'Dropout', 'Gemm', 'Flatten', 'Relu', 'MaxPool', 'BatchNormalization', 'Conv'}
# Union of the ones before - {'Add', 'AveragePool', 'BatchNormalization', 'Concat', 'Conv', 'Dropout', 'Flatten', 'Gemm', 'GlobalAveragePool', 'MaxPool', 'Mul', 'Relu', 'Unsqueeze'}


def pads_from_onnx_to_pydtnn(pads: list[int]) -> tuple[int, int]:  # -> list[tuple[int, int]]:
    # "pads format should be as follow [x1_begin, x2_begin…x1_end, x2_end,…]" from, for example, https://onnx.ai/onnx/operators/onnx__AveragePool.html
    # Onnx: [x1_begin, x2_begin, ..., x1_end, x2_end, ...] ==> "PyDTNN: [(x1_begin, x1_end), (x2_end, x2_begin), ...]"
    # ==> PyDTNN only admits a int or a (vpadding, hpadding) ==> It's assumed that is the first tuple.

    print(f"pads: {pads}")  # TODO: Borrar
    num_pads = len(pads)//2
    _pads = [(0, 0)] * (num_pads)
    for i in range(num_pads):
        _pads[i] = (int(pads[i]), int(pads[i + num_pads]))
    print(f"_pads: {_pads}")  # TODO: Borrar

    return _pads[0]
# --- END pads_from_onnx_to_pydtnn --- #


def not_implemented(name: str) -> Callable:
    # Normal usage of this: switch_pytorch_pydtnn([not_implemented_layer_name])(args)
    def _not_implemented(args: dict[str, Any]) -> None:
        raise NotImplementedError(f"Layer \"{name}\" not implemented - Args received:\n{args} ")
    return _not_implemented
# --- END not_implemented --- #


def switch_onnx_operation_to_pydtnn(name: str) -> Callable[[dict[str, Any]], LayerBase]:
    match name:
        case "Add": return Add
        case "AveragePool": return AveragePool
        case "BatchNormalization": return BatchNormalization
        case "Concat": return Concat
        case "Conv": return Conv
        case "Dropout": return Dropout
        case "Flatten": return Flatten
        case "Gemm": return Gemm
        case "GlobalAveragePool": return GlobalAveragePool
        case "MaxPool": return MaxPool
        case "Mul": return Mul
        case "Relu": return Relu
        case "Unsqueeze": return Unsqueeze
        # Base case:
        case _: return not_implemented(name)
# --- END switch_onnx_operation_to_pydtnn --- #
