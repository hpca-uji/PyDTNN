"""Translation functions to convert PyTorch functional operations into their corresponding PyDTNN layer."""

import logging
from typing import Any

from pydtnn.activations.log_softmax import LogSoftmax
import pydtnn.converters.pytorch2pydtnn.common as cm
from pydtnn.abstract.layerable import Layerable
from pydtnn.activations.log_sigmoid import LogSigmoid
from pydtnn.activations.relu import Relu
from pydtnn.activations.sigmoid import Sigmoid
from pydtnn.activations.softmax import Softmax
from pydtnn.activations.tanh import Tanh
from pydtnn.converters.pytorch2pydtnn.layers import activation
from pydtnn.layers.adaptive_average_pool_2d import AdaptiveAveragePool2D
from pydtnn.layers.addition_block import AdditionBlock
from pydtnn.layers.average_pool_2d import AveragePool2D
from pydtnn.layers.concatenation_block import ConcatenationBlock
from pydtnn.layers.flatten import Flatten

__all__ = (
    "adaptive_avg_pool_2d",
    "add",
    "concat",
    "flatten",
    "log_sigmoid",
    "relu",
    "sigmoid",
    "softmax",
    "tanh",
)

logger = logging.getLogger(__name__)

# - Torch Functions

def adaptive_avg_pool_2d(args: dict[str, str]) -> tuple[AveragePool2D, str]:
    """
    Converts PyTorch adaptive average pooling operation to PyDTNN AdaptiveAveragePool2D layer.

    Args:
        args: Dictionary containing operation parameters and input configuration.

    Returns:
        A tuple containing the initialized AdaptiveAveragePool2D layer and the input layer name.
    """
    # It is not the layer, but the operation itself.
    # from torch.nn.functional import adaptive_avg_pool2d
    # adaptive_avg_pool2d(input: Tensor, output_size: BroadcastingList2[int])

    dict_params = dict()
    # Example: torch.nn.functional.adaptive_avg_pool2d(relu, (1, 1)) | args = 'relu, (1, 1)'
    params: list = args[cm.PARAMETERS].split(cm.ARGS_SEPARATOR)
    # removing the input layer:
    dict_params["input"] = params.pop(
        0
    )  # Situation after operation: [] or ['number'] or ['(number', 'number)']

    # Getting the arguments:
    match len(params):
        case 0:
            params = []
        case 1:
            param = int(params[0])
            params = [param, param]  # Only 1 argument implies the weight and height are the same.
        case _:  # len must be always >= 0
            params = [int(param.replace("(", "").replace(")", "")) for param in params]

    if params:
        dict_params[cm.ARGUMENTS] = {cm.PYTORCH_OUTPUT_SIZE: params}

    return (AdaptiveAveragePool2D(params), dict_params["input"])  # pyright: ignore[reportReturnType,reportArgumentType]


def add(args: dict[str, Any]) -> tuple[AdditionBlock, str]:
    """
    Converts PyTorch addition operation to PyDTNN AdditionBlock layer.

    Args:
        args: Dictionary containing operation parameters and layer graph state.

    Returns:
        A tuple containing the initialized AdditionBlock layer and the input layer name.
    """
    # https://pytorch.org/docs/stable/generated/torch.add.html

    # It should be prepared so the params have the following format: "[layer1,layer2]"
    layer_name: str = args[cm.OPERATION_VAR]
    dict_equivalent_layers = args[cm.EQUIVALENT_LAYERS]
    params = cm.separate_function_params(args[cm.PARAMETERS])

    params = cm.get_equivalent_layer(params, dict_equivalent_layers)
    dict_layers: dict[str, tuple[Layerable, str]] = args[cm.LAYERS]

    list_layers, to_remove, input_layer_name = cm.get_lists_operations_and_outputs(
        dict_layers=dict_layers, layer_inputs=params
    )

    to_remove = set(
        to_remove
    )  # Remove multiple ocurrences of a layer. Consecuence of "get_equivalent_layer".
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


def concat(args: dict[str, Any]) -> tuple[ConcatenationBlock, str]:
    """
    Converts PyTorch concatenation operation to PyDTNN ConcatenationBlock layer.

    Args:
        args: Dictionary containing operation parameters and layer graph state.

    Returns:
        A tuple containing the initialized ConcatenationBlock layer and the input layer name.
    """
    # https://pytorch.org/docs/main/generated/torch.cat.html

    # TODO: es necesario hacer un diccionario que sustituya los parámetros que ya han sido introducidos
    #   por la capa de concatenación/adición.
    #   También hay que haer que solo aparezca una única vez.
    layer_name: str = args[cm.OPERATION_VAR]
    dict_equivalent_layers: dict[str, str] = args[cm.EQUIVALENT_LAYERS]
    parameters: list[str] = args[cm.PARAMETERS].split("],")

    # Since PyDTNN always concatenate in the same dimensions, the rest of the
    # PyTorch parameters can be ignored
    params = parameters.pop(0)
    params = cm.separate_function_params(params)
    params = cm.get_equivalent_layer(params, dict_equivalent_layers)

    dict_layers: dict[str, tuple[Layerable, str]] = args[cm.LAYERS]
    list_layers, to_remove, input_layer_name = cm.get_lists_operations_and_outputs(
        dict_layers=dict_layers, layer_inputs=params
    )

    to_remove = set(
        to_remove
    )  # Remove multiple ocurrences of a layer. Consecuence of "get_equivalent_layer".

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


def flatten(args: dict[str, str]) -> tuple[Flatten, str]:
    """
    Converts PyTorch flatten operation to PyDTNN Flatten layer.

    Args:
        args: Dictionary containing operation parameters.

    Returns:
        A tuple containing the initialized Flatten layer and the input layer name.
    """
    # https://pytorch.org/docs/stable/generated/torch.flatten.html
    # torch.flatten(input, start_dim=0, end_dim=-1)

    def switch(list_params: list[str], dict_params: dict[str, str] = dict()) -> dict[str, str]:
        """Helper to parse flatten parameters recursively."""
        # This is a switch with "fall through".
        # FIXME: this also changes types but typing says otherwise
        match len(list_params):
            case 3:
                var = list_params.pop().split("end_dim=")
                dict_params["end_dim"] = int(var.pop())  # pyright: ignore[reportArgumentType]
                return switch(list_params, dict_params)
            case 2:
                var = list_params.pop().split("start_dim=")
                dict_params["start_dim"] = int(var.pop())  # pyright: ignore[reportArgumentType]
                return switch(list_params, dict_params)
            case 1:
                dict_params["input"] = list_params.pop()
                return switch(list_params, dict_params)
            case _:
                return dict_params

    params = args[cm.PARAMETERS].strip()
    dict_params = switch(params.split(cm.ARGS_SEPARATOR))

    # return (Flatten(**dict_params), dict_params["input"])
    return (Flatten(), dict_params["input"])


# --- Activations --


def log_sigmoid(args: dict[str, Any]) -> tuple[LogSigmoid, str]:
    """
    Converts PyTorch log_sigmoid activation operation to PyDTNN Log layer.

    Args:
        args: Dictionary containing operation parameters.

    Returns:
        A tuple containing the initialized LogSigmoid layer and the input layer name.
    """
    # https://pytorch.org/docs/stable/generated/torch.nn.functional.logsigmoid.html#torch.nn.functional.logsigmoid

    dict_params = dict()

    # Example: torch.nn.functional.relu(features_norm5, inplace = True)
    params = args[cm.PARAMETERS].strip().split("inplace=")
    inplace = bool(params.pop()) if len(params) > 0 else None

    dict_params[cm.ARGUMENTS] = {"input": params[0].split(cm.ARGS_SEPARATOR)[0]}
    if inplace is not None:
        dict_params["inplace"] = inplace

    return (activation.LogSigmoid(**dict_params), dict_params["input"])

def log_softmax(args: dict[str, Any]) -> tuple[LogSoftmax, str]:
    """
    Converts PyTorch log_softmax activation operation to PyDTNN Log layer.

    Args:
        args: Dictionary containing operation parameters.

    Returns:
        A tuple containing the initialized LogSoftmax layer and the input layer name.
    """
    # https://pytorch.org/docs/stable/generated/torch.nn.functional.logsigmoid.html#torch.nn.functional.logsigmoid
    # log_softmax(input, dim=None, _stacklevel=3, dtype=None)

    def switch(list_params: list[str], dict_params: dict[str, str] = dict()) -> dict[str, str]:
        """Helper to parse softmax parameters recursively."""
        # This is a switch with "fall through".
        # FIXME: this also changes types but typing says otherwise
        match len(list_params):
            case 3:
                var = list_params.pop().split("dim=")
                dict_params["end_dim"] = int(var.pop())  # pyright: ignore[reportArgumentType]
                return switch(list_params, dict_params)
            case 2:
                var = list_params.pop().split("dtype=")
                dict_params["start_dim"] = int(var.pop())  # pyright: ignore[reportArgumentType]
                return switch(list_params, dict_params)
            case 1:
                dict_params["input"] = list_params.pop()
                return switch(list_params, dict_params)
            case _:
                return dict_params

    params = args[cm.PARAMETERS].strip()
    dict_params = switch(params.split(cm.ARGS_SEPARATOR))

    return (activation.LogSoftmax(**dict_params), dict_params["input"])  # pyright: ignore[reportArgumentType]

def relu(args: dict[str, str]) -> tuple[Relu, str]:
    """
    Converts PyTorch ReLU activation operation to PyDTNN Relu layer.

    Args:
        args: Dictionary containing operation parameters.

    Returns:
        A tuple containing the initialized Relu layer and the input layer name.
    """

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


def sigmoid(args: dict[str, Any]) -> tuple[Sigmoid, str]:
    """
    Converts PyTorch sigmoid activation operation to PyDTNN Sigmoid layer.

    Args:
        args: Dictionary containing operation parameters.

    Returns:
        A tuple containing the initialized Sigmoid layer and the input layer name.
    """
    # https://pytorch.org/docs/stable/generated/torch.nn.functional.sigmoid.html#torch.nn.functional.sigmoid
    # Not used Pytorch's parameters: inplace.

    dict_params = dict()

    params: list[str] = args[cm.PARAMETERS].split(cm.ARGS_SEPARATOR)
    # removing the input layer:
    dict_params["input"] = params.pop(0)

    return (activation.Sigmoid(**dict_params), dict_params["input"])


def softmax(args: dict[str, Any]) -> tuple[Softmax, str]:
    """
    Converts PyTorch softmax activation operation to PyDTNN Softmax layer.

    Args:
        args: Dictionary containing operation parameters.

    Returns:
        A tuple containing the initialized Softmax layer and the input layer name.
    """
    # https://pytorch.org/docs/stable/generated/torch.nn.functional.softmax.html#torch.nn.functional.softmax
    # softmax(input, dim=None, _stacklevel=3, dtype=None)

    def switch(list_params: list[str], dict_params: dict[str, str] = dict()) -> dict[str, str]:
        """Helper to parse softmax parameters recursively."""
        # This is a switch with "fall through".
        # FIXME: this also changes types but typing says otherwise
        match len(list_params):
            case 3:
                var = list_params.pop().split("dim=")
                dict_params["end_dim"] = int(var.pop())  # pyright: ignore[reportArgumentType]
                return switch(list_params, dict_params)
            case 2:
                var = list_params.pop().split("dtype=")
                dict_params["start_dim"] = int(var.pop())  # pyright: ignore[reportArgumentType]
                return switch(list_params, dict_params)
            case 1:
                dict_params["input"] = list_params.pop()
                return switch(list_params, dict_params)
            case _:
                return dict_params

    params = args[cm.PARAMETERS].strip()
    dict_params = switch(params.split(cm.ARGS_SEPARATOR))

    return (activation.Softmax(**dict_params), dict_params["input"])  # pyright: ignore[reportArgumentType]


def tanh(args: dict[str, Any]) -> tuple[Tanh, str]:
    """
    Converts PyTorch tanh activation operation to PyDTNN Tanh layer.

    Args:
        args: Dictionary containing operation parameters.

    Returns:
        A tuple containing the initialized Tanh layer and the input layer name.
    """
    # https://pytorch.org/docs/stable/generated/torch.nn.functional.tanh.html#torch.nn.functional.tanh
    dict_params = dict()

    params: list[str] = args[cm.PARAMETERS].split(cm.ARGS_SEPARATOR)
    # removing the input layer:
    dict_params["input"] = params.pop(0)

    return (activation.Tanh(**dict_params), dict_params["input"])
