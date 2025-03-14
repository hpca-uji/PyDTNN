# Typing related (or non important) imports
from typing import *
from pydtnn.layers.layer_and_activation_base import LayerAndActivationBase
from torch.nn import Module

# Functionality imports
from pydtnn.translators.pytorch2pydtnn.layers.normalization import *
from pydtnn.translators.pytorch2pydtnn.layers.convolutional import *
from pydtnn.translators.pytorch2pydtnn.layers.activation import *
from pydtnn.translators.pytorch2pydtnn.layers.functions import *
from pydtnn.translators.pytorch2pydtnn.layers.pooling import *
from pydtnn.translators.pytorch2pydtnn.layers.dropout import *
from pydtnn.translators.pytorch2pydtnn.layers.linear import *

# ------------------- #
# ---- CONSTANTS ---- #
# ------------------- #
ARGUMENTS = "arguments"
PARAMETERS = "parameters"
LAYERS = "layers"
EQUIVALENT_LAYERS = "equivalent_layers" # TODO: Set a better name.
OPERATION_VAR = "operation_var" # TODO: Set a better name.

RELU = "relu"
ADP_AVG_POOL = "adaptive_avg_pool2d"

ADD = "add"
CONCAT = "concat"
CAT = "cat"
FLATTEN = "flatten"

ARGS_SEPARATOR = ','
PYTORCH_OUTPUT_SIZE = "output_size"

SPECIAL_CASES = ["torchvision_models_googlenet_GoogLeNetOutputs"]

# TODO: Borrar
MODELO = None
# -> torchvision_models_googlenet_GoogLeNetOutputs: is a named tuple. If both aux layers exist and it is not expected their outputs, the output is only the FC's one.
# 
# ------------------- #

# ------------------- #
# ---- FUNCTIONS ---- #
# ------------------- #
def not_implemented(name: str) -> Callable:
    # Normal usage of this: switch_pytorch_pydtnn([not_implemented_layer_name])(args)
    def _not_implemented(args: Dict[str, Any]) -> None:
        raise NotImplementedError(f"Layer \"{name}\" not implemented - Args received:\n{args} ")
    return _not_implemented
# --- END not_implemented --- # 

def prepare_pydtnn_arguments(arguments: Dict[str, Any], torch_dict_keys: List[str], pydtnn_dict_keys: List[str]) -> Dict[str, Any]:
    #layer_args = dict()
    #for torch_key, pydtnn_key in zip(torch_dict_keys, pydtnn_dict_keys):
    #    if torch_key in arguments:
    #        layer_args[pydtnn_key] = arguments[torch_key]
    #return layer_args
    return {pydtnn_key: arguments[torch_key] for torch_key, pydtnn_key in zip(torch_dict_keys, pydtnn_dict_keys) if torch_key in arguments}
# --- END prepare_pydtnn_arguments --- #

# TODO: Check what to do if it's a call to a torch function (and check torch functions to implement)
def switch_pytorch_pydtnn(name:str) -> Callable[[Dict[str, Any]], LayerAndActivationBase]:
    match name:
        case "Conv2d": return Conv2d
        case "Linear": return Linear
        case "BatchNorm2d": return BatchNorm2d
        case "ReLU": return ReLU
        case "AdaptiveAvgPool2d": return AdaptiveAvgPool2d
        case "AvgPool2d": return AvgPool2d
        case "MaxPool2d": return MaxPool2d
        case "Dropout": return Dropout
        # Not actual PyTorch layers (are torch functions):
        case "Add": return Add # if the constants ADD values are changed, change the case in order to have the same value.
        case "Concat": return Concat # if the constants CONCAT values are changed, change the case in order to have the same value.        
        # Base case:
        case _: return not_implemented(name)
# --- END switch_pytorch_pydtnn --- #

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

# TODO: Check what to do if it's a call to a torch function (and check torch functions to implement)
def function_operation_to_pydtnn(name:str) -> Callable[[Dict[str, Any]], Tuple[LayerAndActivationBase, str]]:
    if ADD in name:
        op = Add
    elif any(pattern in name for pattern in [CONCAT, CAT]):
        op = Concat
    elif FLATTEN in name:
        op = Flatten
    elif RELU in name:
        # It is not the layer, but the operation itself.
        op = relu
    elif ADP_AVG_POOL in name:
        # It is not the layer, but the operation itself.
        op = adaptive_avg_pool_2d
    else:
        op =  not_implemented(name)
    return op
# --- END function_operation_to_pydtnn --- #

def get_lists_operations_and_outputs(dict_layers: Dict[str, Tuple[LayerAndActivationBase, str]], layer_inputs: List[str]) -> Tuple[List[List[LayerAndActivationBase]], List[str], str]:
    # NOTE: It is assumed that the model will by a feed-forward network 
    dict_branch = {}
    # -- Making the "path" of layers for every input -- #
    for inpt in layer_inputs:
        dict_branch[inpt] = dict() 
        input_search = inpt        
        while input_search is not None:
            #operations: {[output_name]: [operation]}
            op, inp = dict_layers[input_search]
            dict_branch[inpt][input_search] = op
            input_search = inp
    # -- Searching the first coincidence -- #
    # Sets are not ordered by insertion ==> keep order with enumerate ==>
    #   ==> braches have different sizes, then the same node may have different order in different branches ==> 
    #       ==> that's true from bottom to top, from top to bottom the "intersection layers" (the ones to be searched) should be at the same position.
    enumerated_reversed_inputs = enumerate(list(dict_branch[layer_inputs[0]].keys())[::-1])
    coincidences = set(enumerated_reversed_inputs)
    for i in range(1, len(layer_inputs)):                
        coincidences = coincidences.intersection(set(enumerate(list(dict_branch[layer_inputs[i]].keys())[::-1])))
    # "Unenumerating" and sorting the intersection, and getting the first coincidence layer.
    #   ==> NOTE: Due the list was sorting in reverse before, now it is necessary to sort it be reverse again (that's why the "-x[0]").
    coincidences = [elem[1] for elem in sorted(coincidences, key=lambda x: -x[0])]
    new_previous_layer = coincidences[0] # new_previous_layer = PyDTNN concat input
    # -- Trimming the dict and storing the data to be returned -- #
    lists_operations: List[LayerAndActivationBase] = list() # List of lists (one per branch)
    lists_outputs: List[str] = list() # List of strings (all branches in one)    
    for inpt in layer_inputs:
        # - Trimming the dict - #
        for coincidence in coincidences:
            del dict_branch[inpt][coincidence] 
        # NOTE: dict_branch[].values() is reversed ==> It is necesarry to un-reverse the layer
        layers = list(dict_branch[inpt].values())[::-1]
        outputs = list(dict_branch[inpt].keys())
        lists_operations.append(layers) # Remember: List of lists (one per branch)
        lists_outputs.extend(outputs)   # Remember: List of strings (all branches in one)
    return (lists_operations, lists_outputs, new_previous_layer)
# --- END get_lists_operations_and_outputs --- #

def separate_function_params(params: str) -> List[str]:
    # Example: '[layer1_0_bn3,layer1_0_downsample_1]'
    params = params.replace('[', '').replace(']', '') # Removing non-useful characters
    params = params.split(',')  
    return [param.strip() for param in params] # Removing spaces
# --- END separate_function_params --- #

# NOTE: This coversor does *not* work in the cases like the following:
# A, B, C, D, E are layers, D and E are layers like concatenation or addition layers.
# A →→ B → D → E
#   ↘→ C →→↑   ↑
#       ↘→→→→→→↑
def get_equivalent_layer(params: List[str], dict_equivalent_layers:Dict[str, str]) -> List[str]:
    # TODO: Check if order is important. If not: dict ==> set
    equivalent_layers = dict()
    for param in params:
        layer = param    
        while layer in dict_equivalent_layers:
            layer = dict_equivalent_layers[layer]
        #else: Nothing special
        equivalent_layers[layer] = None
    return list(equivalent_layers.keys())
# --- END get_equivalent_layer ---#


def print_dict(dictionary: Dict[str, Any], name:str) -> None:
    print(name)
    for k in dictionary.keys():
        if k == "_parameters" and "weight" in dictionary[k]:
            print(f"\t {k}: {dictionary[k]["weight"].shape}")
        elif not k.startswith("_"):
            print(f"\t {k}: {dictionary[k]}")
    print("-----")
# --- END print_dict --- #

# ------------------- #