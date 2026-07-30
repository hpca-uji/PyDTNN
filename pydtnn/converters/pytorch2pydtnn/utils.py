"""Common utilities and mapping functions for converting PyTorch models to PyDTNN."""

import logging
from typing import Any
from collections.abc import Callable

import numpy as np
import torch

from pydtnn.abstract.layerable import Layerable
from pydtnn.converters.pytorch2pydtnn.layers.activation import (LeakyRelu, LogSigmoid, LogSoftmax,
                                                                ReLU, ReLU6, Sigmoid, Softmax, Tanh)
from pydtnn.converters.pytorch2pydtnn.layers.convolutional import Conv2d
from pydtnn.converters.pytorch2pydtnn.layers.dropout import Dropout
from pydtnn.converters.pytorch2pydtnn.layers.functions import (adaptive_avg_pool_2d, add, relu,
                                                               concat, flatten, log_sigmoid, log_softmax,
                                                               sigmoid, softmax, tanh)
from pydtnn.converters.pytorch2pydtnn.layers.linear import Linear
from pydtnn.converters.pytorch2pydtnn.layers.normalization import BatchNorm2d
from pydtnn.converters.pytorch2pydtnn.layers.pooling import AdaptiveAvgPool2d, AvgPool2d, MaxPool2d
from pydtnn.converters.pytorch2pydtnn.layers.utility import Flatten, Identity
from pydtnn.utils.tensor import format_transpose

__all__ = (
    "function_operation_to_pydtnn",
    "get_equivalent_layer",
    "get_lists_operations_and_outputs",
    "not_implemented",
    "prepare_pydtnn_arguments",
    "separate_function_params",
    "switch_operation_symbols",
    "switch_pytorch_pydtnn",
)

logger = logging.getLogger(__name__)

# ---- CONSTANTS ----
ARGUMENTS = "arguments"
PARAMETERS = "parameters"
LAYERS = "layers"
EQUIVALENT_LAYERS = "equivalent_layers"
OPERATION_VAR = "operation_var"
TRANSPOSE_WEIGHTS_LAYERS = [
    "Linear"
]  # There are layers that put the weigths in the correct order. Theese layers doesn't do it.
REMOVE_WIGHTS_DIMENSIONS = [
    ("Conv2d", (0))
]  # Name of the layer, tuple of dimensions/axis to remove.

RELU = "relu"
ADP_AVG_POOL = "adaptive_avg_pool2d"

ADD = "add"
CONCAT = "concat"
CAT = "cat"
FLATTEN = "flatten"
TANH = "tanh"
SOFTMAX = "softmax"
SIGMOID = "sigmoid"
LOG_SIGMOID = "logsigmoid"
LOG_SOFTMAX = "log_softmax"

ARGS_SEPARATOR = ","
PYTORCH_OUTPUT_SIZE = "output_size"

SPECIAL_CASES = ["torchvision_models_googlenet_GoogLeNetOutputs"]
# SPECIAL CASES:
# -> torchvision_models_googlenet_GoogLeNetOutputs: is a "named tuple". If both aux layers exist
#    and it is not expected their outputs, the actual output is only the FC's one.
# END SPECIAL CASES


# ---- FUNCTIONS ----


def not_implemented(name: str) -> Callable:
    """
    Returns a function that raises NotImplementedError when called.

    Args:
        name: The name of the layer or operation that is not implemented.

    Returns:
        A callable that raises an exception.
    """

    # Normal usage of this: switch_pytorch_pydtnn([not_implemented_layer_name])(args)
    def _not_implemented(args: dict[str, Any]) -> None:
        raise NotImplementedError(f"Layer {name} not implemented - Args received:\n{args} ")

    return _not_implemented


def prepare_pydtnn_arguments(
    arguments: dict[str, Any], torch_dict_keys: list[str], pydtnn_dict_keys: list[str]
) -> dict[str, Any]:
    """
    Maps PyTorch argument keys to PyDTNN argument keys.

    Args:
        arguments: Dictionary containing raw arguments.
        torch_dict_keys: List of keys present in the PyTorch arguments.
        pydtnn_dict_keys: List of corresponding keys for PyDTNN.

    Returns:
        A dictionary with mapped keys.
    """
    return {
        pydtnn_key: arguments[torch_key]
        for torch_key, pydtnn_key in zip(torch_dict_keys, pydtnn_dict_keys)
        if torch_key in arguments
    }


def switch_pytorch_pydtnn(name: str) -> Callable[[dict[str, Any]], Layerable]:
    """
    Maps a PyTorch layer name to its corresponding PyDTNN layer class or function.

    Args:
        name: The name of the PyTorch layer.

    Returns:
        The corresponding PyDTNN layer class or function.
    """
    # NOTE: name is the result of torch.nn.[layer]._get_name();
    #   if PyTorch change their layer's names, then it's necessary to change the names here.
    match name:
        case "AdaptiveAvgPool2d":
            return AdaptiveAvgPool2d
        case "AvgPool2d":
            return AvgPool2d
        case "BatchNorm2d":
            return BatchNorm2d
        case "Conv2d":
            return Conv2d
        case "Dropout":
            return Dropout
        case "Linear":
            return Linear
        case "MaxPool2d":
            return MaxPool2d
        case "ReLU":
            return ReLU
        case "ReLU6":
            return ReLU6
        case "LeakyReLU":
            return LeakyRelu
        case "LogSigmoid":
            return LogSigmoid
        case "LogSoftmax":
            return LogSoftmax
        case "Sigmoid":
            return Sigmoid
        case "Softmax":
            return Softmax
        case "Tanh":
            return Tanh
        case "Flatten":
            return Flatten
        case "Identity":
            return Identity

        # Not actual PyTorch layers (are torch functions):
        case "Add":
            # Possible FIXME: if the constants ADD values are changed,
            # change the case in order to have the same value.
            return add  # pyright: ignore[reportReturnType]
        case "Concat":
            # Possible FIXME: if the constants CONCAT values are
            # changed, change the case in order to have the same value.
            return concat  # pyright: ignore[reportReturnType]
        # Base case:
        case _:
            return not_implemented(name)


def switch_operation_symbols(op: str) -> str:
    """
    Maps operator symbols to internal operation constants.

    Args:
        op: The operator symbol (e.g., '+').

    Returns:
        The internal constant string for the operation.
    """
    match op:
        case "+":
            op = ADD
        # Base case:
        case _:
            not_implemented(op)("")
            op = "NOT_IMPLEMENTED"
    return op


def function_operation_to_pydtnn(name: str) -> Callable[[dict[str, Any]], tuple[Layerable, str]]:
    """
    Maps a PyTorch functional operation name to its corresponding PyDTNN function.

    Args:
        name: The name of the functional operation.

    Returns:
        The corresponding PyDTNN function.
    """

    # NOTE: I found impossible to do a switch (match-case) nor a dictionary
    # due the name may be larger than the "key" (e.g.: name =
    # torch.flatten(input, start_dim=0, end_dim=-1); "key" = "flatten")
    if ADP_AVG_POOL in name:
        op = adaptive_avg_pool_2d
    elif LOG_SIGMOID in name:
        op = log_sigmoid
    elif LOG_SOFTMAX in name:
        op = log_softmax
    elif SOFTMAX in name:
        op = softmax
    elif FLATTEN in name:
        op = flatten
    elif SIGMOID in name:
        # NOTE: is important that SIGMOID is after LOG_SIGMOID
        op = sigmoid
    elif RELU in name:
        # It is not the layer, but the relu operation itself.
        op = relu
    elif TANH in name:
        op = tanh
    elif ADD in name:
        op = add
    elif CAT in name:
        op = concat
    # NOTE: If a new function operation handler is implemented, an "elif" must
    # be place before the followin else in order to call the handler of that
    # operation.
    else:
        op = not_implemented(name)
    return op


def get_lists_operations_and_outputs(
    dict_layers: dict[str, tuple[Layerable, str]], layer_inputs: list[str]
) -> tuple[list[list[Layerable]], list[str], str]:
    """
    Traces the network graph to organize operations and outputs for branches.

    Args:
        dict_layers: Dictionary mapping output names to (operation, input_name) tuples.
        layer_inputs: List of input names to trace.

    Returns:
        A tuple containing the list of operations per branch, list of all output names, and the common previous layer.
    """
    # NOTE: It is assumed that the model will by a feed-forward network
    dict_branch = {}

    # -- Making the "path" of layers for every input --

    for inpt in layer_inputs:
        dict_branch[inpt] = dict()
        input_search = inpt
        while input_search is not None:
            # operations: {[output_name]: [operation]}
            op, inp = dict_layers[input_search]
            dict_branch[inpt][input_search] = op
            input_search = inp
        # end while
    # end for

    # -- Searching the first coincidence --

    # NOTE: This is the flow of my thougths regarding the approach:
    #  > Sets are not ordered by insertion ==> keep order with enumerate ==>
    #  > ==> braches have different sizes, then the same node may have different order in different branches ==>
    #  > ==> that's true from bottom to top, but from top to bottom the "intersection layers"
    #           -the ones to be searched- (the ones that coincide in all branches) must be in the same position in every branch.
    enumerated_reversed_inputs = enumerate(list(dict_branch[layer_inputs[0]].keys())[::-1])
    coincidences = set(
        enumerated_reversed_inputs
    )  # NOTE: It is necessary to have a set with elements in order to make an intersection.
    for i in range(1, len(layer_inputs)):
        coincidences = coincidences.intersection(
            set(enumerate(list(dict_branch[layer_inputs[i]].keys())[::-1]))
        )
    # "Unenumerating" and sorting the intersection, and getting the first coincidence layer.
    #   ==> NOTE: Due the list was sorting in reverse before,
    #              now it is necessary to sort it be reverse again (that's why the "-x[0]").
    coincidences = [elem[1] for elem in sorted(coincidences, key=lambda x: -x[0])]
    new_previous_layer = coincidences[0]  # new_previous_layer = PyDTNN concat input

    # -- Trimming the dict and storing the data to be returned --

    lists_operations: list[list[Layerable]] = list()  # list of lists (one list per branch)
    lists_outputs: list[str] = list()  # list of strings (all branches in one list)
    for inpt in layer_inputs:
        # - Trimming the dict -
        for coincidence in coincidences:
            del dict_branch[inpt][coincidence]

        # - Setting the lists of operations and outputs -
        # NOTE: dict_branch[].values() is reversed ==> It is necesarry to unreverse the layer
        layers = list(dict_branch[inpt].values())[::-1]
        outputs = list(dict_branch[inpt].keys())
        lists_operations.append(layers)  # NOTE: Remember, this is a list of lists (one per branch)
        lists_outputs.extend(
            outputs
        )  # NOTE: Remember, this is a list of strings (all branches in one)
    # for inpt in layer_inputs end
    return (lists_operations, lists_outputs, new_previous_layer)


def separate_function_params(params: str) -> list[str]:
    """
    Parses a string representation of parameters into a list of strings.

    Args:
        params: String containing parameters (e.g., '[a,b]').

    Returns:
        A list of parameter strings.
    """
    # Example: '[layer1_0_bn3,layer1_0_downsample_1]'
    params = params.replace("[", "").replace("]", "")  # Removing non-useful characters
    return [param.strip() for param in params.split(",")]  # Removing spaces


# NOTE: This coversor does *not* work in the cases like the following:
# A, B, C, D, E are layers, D and E are layers like concatenation or addition layers.
# A →→ B → D → E
#   ↘→ C →→↑   ↑
#       ↘→→→→→→↑


def get_equivalent_layer(params: list[str], dict_equivalent_layers: dict[str, str]) -> list[str]:
    """
    Resolves equivalent layer names based on a mapping dictionary.

    Args:
        params: List of layer names.
        dict_equivalent_layers: Dictionary mapping original names to equivalent names.

    Returns:
        A list of resolved equivalent layer names.
    """
    equivalent_layers = dict()
    for param in params:
        layer = param
        while layer in dict_equivalent_layers:
            layer = dict_equivalent_layers[layer]
        equivalent_layers[layer] = None
    return list(equivalent_layers.keys())


def set_initializer_with_pytorch_values(
    state_dict: dict[str, Any],
    vars_initiaizers_transpose: dict[str, tuple[str, None | tuple[str, str]]] = {
        "weight": ("weights_initializer", None),
        "bias": ("biases_initializer", None),
        "running_mean": ("running_mean_initializer", None),
        "running_var": ("running_var_initializer", None),
    },
) -> dict[str, Any]:
    """Function to set the value returned by the initializers of the layer's weight, bias, etc.

    Args:
        state_dict (dict[str, Any]): A dictionary with the values.
        vars_and_initiaizers (dict[str, tuple[str, None | tuple[str, str]]]): Key (str): the name of the variable.
            Value[0] (str): The name of the initalizer's fuction.
            Value[1] (None | tuple[str, str]): None if it's not necessary to transpose,
                                               tuple([origin shape], [new shape]) if it's necessary to transpose.

    Returns:
        A dict[str, Any] where the Key is the initializer's fuction name, and the value is the initializer's function.
    """
    dict_initalizers: dict[str, Any] = dict()

    for variable in vars_initiaizers_transpose.keys():
        # There are layers without weight nor biases
        if variable in state_dict:
            initalizer_name = vars_initiaizers_transpose[variable][0]
            transpose_values = vars_initiaizers_transpose[variable][1]

            torch_value: torch.Tensor | None = state_dict[variable]
            if torch_value is None:
                continue

            value_to_set: np.ndarray = torch_value.numpy(force=True).copy()
            # NOTE: There are some layers (like the fully connected) where the shape
            # in PyDTNN is the transpose of the PyTorch's one.
            if transpose_values is not None:
                value_to_set = format_transpose(
                    value_to_set, transpose_values[0], transpose_values[1]
                )

            def pytorch_value_initializer(
                shape: tuple,
                dtype: np.ndarray,
                random: np.random.Generator = None,  # pyright: ignore[reportArgumentType]
                pytorch_value_to_set: np.ndarray = value_to_set,
                **kwargs_to_ignore: Any,
            ) -> np.ndarray:
                # NOTE [IMPORTANT]: Regarding "pytorch_value_to_set = value_to_set".
                # NOTE If "value_to_set" is directly set as the returned value ("return value_to_set"),
                #       for some reason the return will be a reference to "value_to_set" instead of
                #       the "value_to_set"'s value (that is a reference to the layer's PyTorch's value_to_set),
                #       so, since this is in a for loop and this function (weights_initializer) is called
                #       in some step after the loop, every layer will have the last iteration's "value_to_set" values
                #       a reference to the last layer value_to_set- instead of a reference to their
                #       respective layer value_to_set.
                #       In this way "pytorch_value_to_set" has the copy of "value_to_set"'s values
                #       (that, as said before, is a reference to the layer's value_to_set) of that iteration.
                return pytorch_value_to_set.astype(dtype=dtype, order="C", copy=False)

            dict_initalizers[initalizer_name] = pytorch_value_initializer
    return dict_initalizers
