# _______________________________________________________________________________________________________________
# In this file must be implemented only the translation of PyTorch functions to its PyDTNN equivalent.
# _______________________________________________________________________________________________________________

# Typing related (or non important) imports
from typing import Dict, Any, Tuple, List
from pydtnn.layer_base import LayerBase

# Functionality imports
from pydtnn.layers.average_pool_2d import AveragePool2D
from pydtnn.layers.adaptive_average_pool_2d import AdaptiveAveragePool2D
from pydtnn.layers.addition_block import AdditionBlock
from pydtnn.layers.concatenation_block import ConcatenationBlock
from pydtnn.layers.flatten import Flatten
from pydtnn.activations.log import Log
from pydtnn.activations.relu import Relu
from pydtnn.activations.sigmoid import Sigmoid
from pydtnn.activations.softmax import Softmax
from pydtnn.activations.tanh import Tanh
import pydtnn.converters.pytorch2pydtnn.common as cm
from pydtnn.converters.pytorch2pydtnn.layers import activation

# ------------------ #
# - Torch Functions  #
# ------------------ #


def adaptive_avg_pool_2d(args: Dict[str, str]) -> Tuple[AveragePool2D, str]:
    # It is not the layer, but the operation itself.
    # from torch.nn.functional import adaptive_avg_pool2d
    # adaptive_avg_pool2d(input: Tensor, output_size: BroadcastingList2[int])

    dict_params = dict()
    # Example: torch.nn.functional.adaptive_avg_pool2d(relu, (1, 1)) | args = 'relu, (1, 1)'
    params: List[str] = args[cm.PARAMETERS].split(cm.ARGS_SEPARATOR)
    # removing the input layer:
    dict_params["input"] = params.pop(0)  # Situation after operation: [] or ['number'] or ['(number', 'number)']

    # Getting the arguments:
    match len(params):
        case 0:
            params = None
        case 1:
            param = int(params[0])
            params = [param, param]  # Only 1 argument implies the weight and height are the same.
        case greater_than_1:  # len must be always >= 0
            params = [int(param.replace('(', '').replace(')', '')) for param in params]

    if params is not None:
        dict_params[cm.ARGUMENTS] = {cm.PYTORCH_OUTPUT_SIZE: params}

    return (AdaptiveAveragePool2D(dict_params), dict_params["input"])


def add(args: Dict[str, Any]) -> Tuple[AdditionBlock, str]:
    # https://pytorch.org/docs/stable/generated/torch.add.html

    # It should be prepared so the params have the following format: "[layer1,layer2]"
    layer_name: str = args[cm.OPERATION_VAR]
    dict_equivalent_layers = args[cm.EQUIVALENT_LAYERS]
    params = cm.separate_function_params(args[cm.PARAMETERS])

    params = cm.get_equivalent_layer(params, dict_equivalent_layers)
    dict_layers: Dict[str, Tuple[LayerBase, str]] = args[cm.LAYERS]

    list_layers, to_remove, input_layer_name = cm.get_lists_operations_and_outputs(dict_layers=dict_layers, layer_inputs=params)

    to_remove = set(to_remove)  # Remove multiple ocurrences of a layer. Consecuence of "get_equivalent_layer".
    # The removed layers will be accesed through the AdditionBlock.
    for elem in to_remove:
        del dict_layers[elem]
    # The equivalences dictionary values are set
    for elem in params:
        dict_equivalent_layers[elem] = layer_name
    # NOTE: Not always "params == to_remove"
    for elem in to_remove:
        dict_equivalent_layers[elem] = layer_name

    # AdditionBlock expects every "branch" (layer list) as a different argument.
    return (AdditionBlock(*list_layers), input_layer_name)


def concat(args: Dict[str, Any]) -> Tuple[ConcatenationBlock, str]:
    # https://pytorch.org/docs/main/generated/torch.cat.html

    # TODO: es necesario hacer un diccionario que sustituya los parámetros que ya han sido introducidos por la capa de concatenación/adición.
    # También hay que haer que solo aparezca una única vez.
    layer_name: str = args[cm.OPERATION_VAR]
    dict_equivalent_layers: Dict[str, str] = args[cm.EQUIVALENT_LAYERS]
    parameters: List[str] = args[cm.PARAMETERS].split("],")

    params = parameters.pop(0)  # Since PyDTNN always concatenate in the same dimensions, the rest of the PyTorch parameters can be ignored
    params = cm.separate_function_params(params)
    params = cm.get_equivalent_layer(params, dict_equivalent_layers)

    dict_layers: Dict[str, Tuple[LayerBase, str]] = args[cm.LAYERS]
    list_layers, to_remove, input_layer_name = cm.get_lists_operations_and_outputs(dict_layers=dict_layers, layer_inputs=params)

    to_remove = set(to_remove)  # Remove multiple ocurrences of a layer. Consecuence of "get_equivalent_layer".

    # The removed layers will be accesed through the ConcatenationBlock.
    for elem in to_remove:
        del dict_layers[elem]
    # The equivalences dictionary values are set
    for elem in params:
        dict_equivalent_layers[elem] = layer_name
    # NOTE: Not always "params == to_remove"
    for elem in to_remove:
        dict_equivalent_layers[elem] = layer_name

    # ConcatenationBlock expects every "branch" (layer list) as a different argument.
    return (ConcatenationBlock(*list_layers), input_layer_name)


def flatten(args: Dict[str, str]) -> Tuple[Flatten, str]:
    # https://pytorch.org/docs/stable/generated/torch.flatten.html
    # torch.flatten(input, start_dim=0, end_dim=-1)

    def switch(list_params: List[str], dict_params: Dict[str, str] = dict()) -> Dict[str, str]:
        # This is a switch with "fall through".
        match len(list_params):
            case 3:
                var = list_params.pop().split("end_dim=")
                dict_params["end_dim"] = int(var.pop())
                # // fall through
                return switch(list_params, dict_params)
            case 2:
                var = list_params.pop().split("start_dim=")
                dict_params["start_dim"] = int(var.pop())
                # // fall through
                return switch(list_params, dict_params)
            case 1:
                dict_params["input"] = list_params.pop()
                # // fall through
                return switch(list_params, dict_params)
            case _:
                return dict_params
    # - END switch - #
    params = args[cm.PARAMETERS].strip()
    dict_params = switch(params.split(cm.ARGS_SEPARATOR))

    return (Flatten(dict_params), dict_params["input"])
# ------------------ #

# ------------------ #
# --- Activations -- #
# ------------------ #


def log(args: Dict[str, Any]) -> Tuple[Log, str]:
    # https://pytorch.org/docs/stable/generated/torch.nn.functional.logsigmoid.html#torch.nn.functional.logsigmoid

    dict_params = dict()

    # Example: torch.nn.functional.relu(features_norm5, inplace = True)
    params = args[cm.PARAMETERS].strip().split("inplace=")
    inplace = bool(params.pop()) if len(params) > 0 else None

    dict_params[cm.ARGUMENTS] = {"input": params[0].split(cm.ARGS_SEPARATOR)[0]}
    if inplace is not None:
        dict_params["inplace"] = inplace

    return (activation.LogSigmoid(**dict_params), dict_params["input"])


def relu(args: Dict[str, str]) -> Tuple[Relu, str]:

    # https://pytorch.org/docs/stable/generated/torch.nn.functional.relu.html#torch.nn.functional.relu
    # It is not the layer, but the operation itself.
    # from torch.nn.functional import relu
    # relu(input: Tensor, inplace: bool = False)

    dict_params = dict()

    # Example: torch.nn.functional.relu(features_norm5, inplace = True)
    params = args[cm.PARAMETERS].strip().split("inplace=")
    inplace = bool(params.pop()) if len(params) > 0 else None

    dict_params[cm.ARGUMENTS] = {"input": params[0].split(cm.ARGS_SEPARATOR)[0]}
    if inplace is not None:
        dict_params["inplace"] = inplace

    return (activation.ReLU(dict_params), dict_params[cm.ARGUMENTS]["input"])


def sigmoid(args: Dict[str, Any]) -> Tuple[Sigmoid, str]:
    # https://pytorch.org/docs/stable/generated/torch.nn.functional.sigmoid.html#torch.nn.functional.sigmoid
    # Not used Pytorch's parameters: inplace.

    dict_params = dict()

    params: List[str] = args[cm.PARAMETERS].split(cm.ARGS_SEPARATOR)
    # removing the input layer:
    dict_params["input"] = params.pop(0)

    return (activation.Sigmoid(**dict_params), dict_params["input"])


def softmax(args: Dict[str, Any]) -> Tuple[Softmax, str]:
    # https://pytorch.org/docs/stable/generated/torch.nn.functional.softmax.html#torch.nn.functional.softmax
    # softmax(input, dim=None, _stacklevel=3, dtype=None)

    def switch(list_params: List[str], dict_params: Dict[str, str] = dict()) -> Dict[str, str]:
        # This is a switch with "fall through".
        match len(list_params):
            case 3:
                var = list_params.pop().split("dim=")
                dict_params["end_dim"] = int(var.pop())
                # // fall through
                return switch(list_params, dict_params)
            case 2:
                var = list_params.pop().split("dtype=")
                dict_params["start_dim"] = int(var.pop())
                # // fall through
                return switch(list_params, dict_params)
            case 1:
                dict_params["input"] = list_params.pop()
                # // fall through
                return switch(list_params, dict_params)
            case _:
                return dict_params
    # - END switch - #
    params = args[cm.PARAMETERS].strip()
    dict_params = switch(params.split(cm.ARGS_SEPARATOR))

    return (activation.Softmax(**dict_params), dict_params["input"])


def tanh(args: Dict[str, Any]) -> Tuple[Tanh, str]:
    # https://pytorch.org/docs/stable/generated/torch.nn.functional.tanh.html#torch.nn.functional.tanh
    dict_params = dict()

    params: List[str] = args[cm.PARAMETERS].split(cm.ARGS_SEPARATOR)
    # removing the input layer:
    dict_params["input"] = params.pop(0)

    return (activation.Tanh(**dict_params), dict_params["input"])

# ------------------ #
