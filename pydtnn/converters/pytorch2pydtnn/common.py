# Typing related (or non important) imports
from typing import Dict, Any, Callable, Tuple, List
from pydtnn.converters.pytorch2pydtnn.layers.activation import LeakyReLU, LogSigmoid, ReLU, ReLU6, Sigmoid, Softmax, Tanh
from pydtnn.converters.pytorch2pydtnn.layers.convolutional import Conv2d
from pydtnn.converters.pytorch2pydtnn.layers.dropout import Dropout
from pydtnn.converters.pytorch2pydtnn.layers.functions import adaptive_avg_pool_2d, add, concat, flatten, relu, sigmoid, softmax, log, tanh
from pydtnn.converters.pytorch2pydtnn.layers.linear import Linear
from pydtnn.converters.pytorch2pydtnn.layers.normalization import BatchNorm2d
from pydtnn.converters.pytorch2pydtnn.layers.pooling import AdaptiveAvgPool2d, AvgPool2d, MaxPool2d
from pydtnn.converters.pytorch2pydtnn.layers.utility import Flatten
from pydtnn.layer_base import LayerBase

# Functionality imports


# ------------------- #
# ---- CONSTANTS ---- #
# ------------------- #
ARGUMENTS = "arguments"
PARAMETERS = "parameters"
LAYERS = "layers"
EQUIVALENT_LAYERS = "equivalent_layers"
OPERATION_VAR = "operation_var"
TRANSPOSE_WEIGHTS_LAYERS = ["Linear"]  # There are layers that put the weigths in the correct order. Theese layers doesn't do it.
REMOVE_WIGHTS_DIMENSIONS = [("Conv2d", (0))]  # Name of the layer, tuple of dimensions/axis to remove.

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

ARGS_SEPARATOR = ','
PYTORCH_OUTPUT_SIZE = "output_size"

SPECIAL_CASES = ["torchvision_models_googlenet_GoogLeNetOutputs"]
# SPECIAL CASES:
# -> torchvision_models_googlenet_GoogLeNetOutputs: is a "named tuple". If both aux layers exist and it is not expected their outputs, the actual output is only the FC's one.
# END SPECIAL CASES

# -- END CONSTANTS -- #
# ------------------- #

# ------------------- #
# ---- FUNCTIONS ---- #
# ------------------- #


def not_implemented(name: str) -> Callable:
    # Normal usage of this: switch_pytorch_pydtnn([not_implemented_layer_name])(args)
    def _not_implemented(args: Dict[str, Any]) -> None:
        raise NotImplementedError(f"Layer \"{name}\" not implemented - Args received:\n{args} ")
    return _not_implemented


def prepare_pydtnn_arguments(arguments: Dict[str, Any], torch_dict_keys: List[str], pydtnn_dict_keys: List[str]) -> Dict[str, Any]:
    return {pydtnn_key: arguments[torch_key] for torch_key, pydtnn_key in zip(torch_dict_keys, pydtnn_dict_keys) if torch_key in arguments}


def switch_pytorch_pydtnn(name: str) -> Callable[[Dict[str, Any]], LayerBase]:
    # NOTE: name is the result of torch.nn.[layer]._get_name();
    #   if PyTorch change their layer's names, then it's necessary to change the names here.
    match name:
        case "AdaptiveAvgPool2d": return AdaptiveAvgPool2d
        case "AvgPool2d": return AvgPool2d
        case "BatchNorm2d": return BatchNorm2d
        case "Conv2d": return Conv2d
        case "Dropout": return Dropout
        case "Linear": return Linear
        case "MaxPool2d": return MaxPool2d
        case "ReLU": return ReLU
        case "ReLU6": return ReLU6
        case "LeakyReLU": return LeakyReLU
        case "LogSigmoid": return LogSigmoid
        case "Sigmoid": return Sigmoid
        case "Softmax": return Softmax
        case "Tanh": return Tanh
        case "Flatten": return Flatten

        # Not actual PyTorch layers (are torch functions):
        case "Add": return add  # Possible FIXME: if the constants ADD values are changed, change the case in order to have the same value.
        case "Concat": return concat  # Possible FIXME: if the constants CONCAT values are changed, change the case in order to have the same value.
        # Base case:
        case _: return not_implemented(name)


def switch_operation_symbols(op: str) -> str:
    match op:
        case "+":
            op = ADD
        # Base case:
        case _:
            not_implemented(op)("")
            op = "NOT_IMPLEMENTED"
    return op
# --- switch_operation_symbols --- #


def function_operation_to_pydtnn(name: str) -> Callable[[Dict[str, Any]], Tuple[LayerBase, str]]:

    # NOTE: I found impossible to do a switch (match-case) nor a dictionary due the name may be larger than the "key" (e.g.: name = torch.flatten(input, start_dim=0, end_dim=-1); "key" = "flatten")
    if ADD in name:
        op = add
    elif any(pattern in name for pattern in [CONCAT, CAT]):
        op = concat
    elif FLATTEN in name:
        op = flatten
    elif RELU in name:
        # It is not the layer, but the relu operation itself.
        op = relu
    elif ADP_AVG_POOL in name:
        op = adaptive_avg_pool_2d
    elif LOG_SIGMOID in name:
        op = log
    elif SIGMOID in name:
        # NOTE: is important that SIGMOID is after LOG_SIGMOID
        op = sigmoid
    elif SOFTMAX in name:
        op = softmax
    elif TANH in name:
        op = tanh
    # NOTE: If a new function operation handler is implemented, an "elif" must be place before the followin else in order to call the handler of that operation.
    else:
        op = not_implemented(name)
    return op


def get_lists_operations_and_outputs(dict_layers: Dict[str, Tuple[LayerBase, str]], layer_inputs: List[str]) -> Tuple[List[List[LayerBase]], List[str], str]:
    # NOTE: It is assumed that the model will by a feed-forward network
    dict_branch = {}

    # -- Making the "path" of layers for every input -- #

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

    # -- Searching the first coincidence -- #

    # NOTE: This is the flow of my thougths regarding the approach:
    #  > Sets are not ordered by insertion ==> keep order with enumerate ==>
    #  > ==> braches have different sizes, then the same node may have different order in different branches ==>
    #  > ==> that's true from bottom to top, but from top to bottom the "intersection layers" -the ones to be searched- (the ones that coincide in all branches) must be in the same position in every branch.
    enumerated_reversed_inputs = enumerate(list(dict_branch[layer_inputs[0]].keys())[::-1])
    coincidences = set(enumerated_reversed_inputs)  # NOTE: It is necessary to have a set with elements in order to make an intersection.
    for i in range(1, len(layer_inputs)):
        coincidences = coincidences.intersection(set(enumerate(list(dict_branch[layer_inputs[i]].keys())[::-1])))
    # "Unenumerating" and sorting the intersection, and getting the first coincidence layer.
    #   ==> NOTE: Due the list was sorting in reverse before, now it is necessary to sort it be reverse again (that's why the "-x[0]").
    coincidences = [elem[1] for elem in sorted(coincidences, key=lambda x: -x[0])]
    new_previous_layer = coincidences[0]  # new_previous_layer = PyDTNN concat input

    # -- Trimming the dict and storing the data to be returned -- #

    lists_operations: List[LayerBase] = list()  # List of lists (one list per branch)
    lists_outputs: List[str] = list()  # List of strings (all branches in one list)
    for inpt in layer_inputs:
        # - Trimming the dict - #
        for coincidence in coincidences:
            del dict_branch[inpt][coincidence]
        # ----

        # - Setting the lists of operations and outputs - #
        # NOTE: dict_branch[].values() is reversed ==> It is necesarry to unreverse the layer
        layers = list(dict_branch[inpt].values())[::-1]
        outputs = list(dict_branch[inpt].keys())
        lists_operations.append(layers)  # NOTE: Remember, this is a list of lists (one per branch)
        lists_outputs.extend(outputs)  # NOTE: Remember, this is a list of strings (all branches in one)
        # ----
    # for inpt in layer_inputs end
    return (lists_operations, lists_outputs, new_previous_layer)


def separate_function_params(params: str) -> List[str]:
    # Example: '[layer1_0_bn3,layer1_0_downsample_1]'
    params = params.replace('[', '').replace(']', '')  # Removing non-useful characters
    params = params.split(',')
    return [param.strip() for param in params]  # Removing spaces

# NOTE: This coversor does *not* work in the cases like the following:
# A, B, C, D, E are layers, D and E are layers like concatenation or addition layers.
# A →→ B → D → E
#   ↘→ C →→↑   ↑
#       ↘→→→→→→↑


def get_equivalent_layer(params: List[str], dict_equivalent_layers: Dict[str, str]) -> List[str]:
    equivalent_layers = dict()
    for param in params:
        layer = param
        while layer in dict_equivalent_layers:
            layer = dict_equivalent_layers[layer]
        equivalent_layers[layer] = None
    return list(equivalent_layers.keys())

# -- END FUNCTIONS -- #
# ------------------- #
