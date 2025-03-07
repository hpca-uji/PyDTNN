# _______________________________________________________________________________________________________________
# In this file must be implemented only the translation of PyTorch functions to its PyDTNN equivalent.
# _______________________________________________________________________________________________________________

# Typing related (or non important) imports
from typing import *
from inspect import stack # This is only in order to get the function's name
from pydtnn import activations

# Functionality imports
from pydtnn import layers
import pydtnn.translators.pytorch2pydtnn.constats as cons
from pydtnn.translators.pytorch2pydtnn.layers.activation import *
from pydtnn.translators.pytorch2pydtnn.layers.pooling import *

# ------------------ #
# - Torch Functions  #
# ------------------ #
# TODO: Check how to do this well.
def Add(args: Dict[str, Any]) -> Tuple[layers.AdditionBlock, str]:
    # https://pytorch.org/docs/stable/generated/torch.add.html      
    print(f"Layer: {stack()[0].function}")
    
    # It should be prepared so the params have the following format: "[layer1,layer2]"
    params = cons.separate_function_params(args[cons.PARAMETERS])
    dict_layers:Dict[str, Tuple[LayerAndActivationBase, str]] = args[cons.LAYERS]

    list_layers, to_remove, input_layer_name = cons.get_lists_operations_and_outputs(dict_layers=dict_layers, layer_inputs=params)

    # The removed layers will be accesed through the AdditionBlock.
    for elem in to_remove:
        del dict_layers[elem]

    # AdditionBlock expects every "branch" (layer list) as a different argument.
    return (layers.AdditionBlock(*list_layers), input_layer_name)
# --- END Add --- #

def Concat(args: Dict[str, Any]) -> Tuple[layers.ConcatenationBlock, str]:
    # https://pytorch.org/docs/main/generated/torch.cat.html
    
    print(f"Layer: {stack()[0].function}")

    parameters:List[str] = args[cons.PARAMETERS].split("],")
    params = parameters.pop(0) # Since PyDTNN always concatenate in the same dimensions, the rest of the PyTorch parameters can be ignored    
    params = cons.separate_function_params(params)
    
    dict_layers:Dict[str, Tuple[LayerAndActivationBase, str]] = args[cons.LAYERS]
    list_layers, to_remove, input_layer_name = cons.get_lists_operations_and_outputs(dict_layers=dict_layers, layer_inputs=params)
    # The removed layers will be accesed through the AdditionBlock.
    for elem in to_remove:
        del dict_layers[elem]

    # ConcatenationBlock expects every "branch" (layer list) as a different argument.
    return (layers.ConcatenationBlock(*list_layers), input_layer_name)
# --- END Concat --- #

def Flatten(args: Dict[str, str]) -> Tuple[layers.Flatten, str]:
    # https://pytorch.org/docs/stable/generated/torch.flatten.html
    #from torch import flatten
    # torch.flatten(input, start_dim=0, end_dim=-1)

    def switch(list_params: List[str], dict_params: Dict[str, str] = dict()) -> Dict[str, str]:
        match len(list_params):
            case 3 :
                    var = list_params.pop().split("end_dim=")
                    dict_params["end_dim"] = int(var.pop())
                    # // fall through
                    return switch(list_params, dict_params)
            case 2 :
                    var = list_params.pop().split("start_dim=")
                    dict_params["start_dim"] = int(var.pop())
                    # // fall through
                    return switch(list_params, dict_params)
            case 1 :
                    dict_params["input"] = list_params.pop()
                    # // fall through
                    return switch(list_params, dict_params)
            case _:
                return dict_params
    # - END switch - #

    print(f"Layer: {stack()[0].function}")
    params = args[cons.PARAMETERS].strip()
    dict_params = switch(params.split(cons.ARGS_SEPARATOR))
        
    return (layers.Flatten(), dict_params["input"])
# --- END Concat --- #


def relu(args: Dict[str, str]) -> Tuple[activations.Relu, str]:
    # It is not the layer, but the operation itself.
    # from torch.nn.functional import relu
    # relu(input: Tensor, inplace: bool = False)
    
    print(f"Layer: {stack()[0].function}")

    dict_params = dict()

    # Example: torch.nn.functional.relu(features_norm5, inplace = True)
    params = args[cons.PARAMETERS].strip().split("inplace=")
    inplace = bool(params.pop()) if len(params) > 0 else None

    dict_params["input"] = params[0].split(cons.ARGS_SEPARATOR)[0]
    if inplace is not None:
        dict_params["inplace"] = inplace

    return (ReLU(dict_params), dict_params["input"])

def adaptive_avg_pool_2d(args: Dict[str, str]) -> Tuple[layers.AveragePool2D, str]:
    # It is not the layer, but the operation itself.
    # from torch.nn.functional import adaptive_avg_pool2d
    # adaptive_avg_pool2d(input: Tensor, output_size: BroadcastingList2[int])

    print(f"Layer: {stack()[0].function}")
    dict_params = dict()
    # Example: torch.nn.functional.adaptive_avg_pool2d(relu, (1, 1)) | args = 'relu, (1, 1)'
    params:List[str] = args[cons.PARAMETERS].split(cons.ARGS_SEPARATOR)
    # removing the input layer:
    dict_params["input"] = params.pop(0) # Situation after operation: [] or ['number'] or ['(number', 'number)']
    
    # Getting the arguments:
    match len(params):
        case 0:
            params = None
        case 1:
            params = int(params[0])
            params = (params, params) # Only 1 argument implies the weight and height are the same.
        case greater_than_1: # len must be always >= 0
            params = [int(param.replace('(', '').replace(')', '')) for param in params]

    if params != None:
        dict_params[cons.ARGUMENTS] = {cons.PYTORCH_OUTPUT_SIZE: params}

    return (AdaptiveAvgPool2d(dict_params), dict_params["input"])
# ------------------ #
