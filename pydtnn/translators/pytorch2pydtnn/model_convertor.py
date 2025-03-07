# Typing related
from typing import *
from pydtnn.layers.layer_and_activation_base import LayerAndActivationBase
import numpy as np

# Operations/transformations related
import torch
from pydtnn.model import Model as PyDTNN_Model
from pydtnn.layers import Input
from pydtnn.utils import PYDTNN_TENSOR_FORMAT_NCHW, PYDTNN_TENSOR_FORMAT_NHWC
import pydtnn.translators.pytorch2pydtnn.constats as cons

def load_layers(model:PyDTNN_Model, layers: List[LayerAndActivationBase]) -> None:

    # TODO: Check if there are more operations to do.
    #   If not ==> Move to the main function.
    for layer in layers:
        print(layer)
        model.add(layer)
# --- END load_layers --- #

def get_model_layers(model:torch.nn.Module, name:str = "self") -> Dict[str, torch.nn.Module]:
    # Recursive function to get the models without containers modules.    
    def _get_model_layers(model:torch.nn.Module, name:str, dict_modules:Dict[str, torch.nn.Module]):
        # The recursive function.
        children = list(model.named_children())
        if len(children) > 0:
            for nom, module in children:
                _get_model_layers(model=module, name=".".join([name, nom]), dict_modules=dict_modules)                
        else:
            dict_modules[name] = model            
    #-- END _get_model_layers--#
    dict_modules = {}
    _get_model_layers(model=model, name=name, dict_modules=dict_modules)
    return dict_modules
# --- END get_model_layers --- #

def extract_layers_relations(model:torch.nn.Module) -> Dict[str, Tuple[Union[str|torch.nn.Module], str]]:
    # TODO: Search the way "torch.fx.symbolic_trace" generates ".code" and not extracting the data from a
    # graph: torch.fx.GraphModule
    graph = torch.fx.symbolic_trace(model)

    # {[output's variable name]: Tuple([string with operation name or the layer object], [string with the args])}
    relations_dic = dict()

    # -- CONSTANTS -- #
    BY_LINES = "\n"
    PSEUDO_INDENTATION = " "
    FIRST_LINE = "forward"
    LAST_LINE = "return"

    SEPARATOR_FUNCTION_VALUE =  ";"
    SEPARATOR_ASSIGNATION = " = " 
    PARAMETERS_BEGINING = "("    
    PARAMETER_ENDING = ")"
    LIST_START = '['
    LIST_SEPARATOR = ','
    LIST_END = ']'
    OPERATION_SEPARATOR = " " # It is expected that the operator is always between spaces (example: "a + b").

    MODEL_LAYER_REQ = "self" # It is a (almost) "necessary" (but not "enough") evidence that, if the line has "self" it is a layer.
    MODEL_FUNCT_ARG_NAME = "model" # NOTE: "model" is the name of the function argument. If it change, it is necessary to change it here.
    TORCH_LAYER_REQ = "torch.nn.functional."
    TORCH_FUNC_REQ = "torch."
    # -- END CONSTANTS -- #

    for line in filter(lambda x: not(FIRST_LINE in x or LAST_LINE in x) , 
                       filter(lambda x: len(x)!=0, 
                        [elem.lstrip(PSEUDO_INDENTATION) for elem in graph.code.split(BY_LINES)])):
        # NOTE: seems that there are situations that the line does not have the value.
        line = line.split(SEPARATOR_FUNCTION_VALUE)[0] # [line, debug's input's value]            
        operation = line.split(SEPARATOR_ASSIGNATION)  # [output, function+args]
        if len(operation) > 2:
            # Case: When it is a call to a function with a keyword. Example: "cat = torch.concatenate([var], axis = 1)"
            output_var = operation.pop(0)
            operation = "=".join(operation) # The spaces are removed to make easier a following step.
        else:
            # Normal case. Example: conv1 = self.conv1(x)
            output_var, operation = operation
        # Now we have split the _output's variable_ and the operation.
        # We want to separate the arguments from the function in order to get the layer and the relations with the previous layers.
        operation = operation.split(PARAMETERS_BEGINING) # [function, ...n..., function, args)]

        func = None # It will be assigned in the following if-else statement
        if len(operation) > 1:
            # Normal case. Examples: 'getattr(self.layer1, "2").bn1(layer1_2_conv1)', 'self.avgpool(features_36)'       
            if any(MODEL_LAYER_REQ in part for part in operation):                
                # Case: 'getattr(self.layer1, "2").bn1(layer1_2_conv1)'
                args = operation.pop().replace(PARAMETER_ENDING, "") # [function, ...n..., function], args
                operation = PARAMETERS_BEGINING.join(operation) # Reasembling the operation without the arguments.
                operation = operation.replace(MODEL_LAYER_REQ, MODEL_FUNCT_ARG_NAME) 
                func = eval(operation) # Getting the layer object.
            else: 
                # Cases: function or layer not defined at model's object's constructor                
                # TORCH_LAYER_REQ --> Case: layer not defined at model's object's constructor
                # Example: "adaptive_avg_pool2d = torch.nn.functional.adaptive_avg_pool2d(relu, (1, 1))" ==>
                # ==> operation = "torch.nn.functional.adaptive_avg_pool2d", args = "relu, (1, 1)"
                # NOTE: The first argument is always a previous layer
                # --
                # TORCH_FUNC_REQ --> Case: function. Example: torch.cat()
                _operation = operation.pop(0) # _operation = something like "torch.cat"; operation= [arg1, arg2) arg3 etc.)] [list[str]]
                args = PARAMETERS_BEGINING.join(operation)[:-1] # _operation = "torch.cat"; operation= arg1 (arg2) arg3 etc. [str] | [:-1] to remove the final ")"
                operation = _operation

                if operation in cons.SPECIAL_CASES:
                    # TODO [possible future FIXME]: See what to do with the special cases.
                    continue

                for pattern in [TORCH_LAYER_REQ, TORCH_FUNC_REQ]:
                    if pattern in operation:
                        func = operation.replace(pattern, "") #operation = "adaptive_avg_pool2d" | "cat"
                        break
                    # else: Never happens. One of the patterns *must* be in operation
        else:          
            # Case "operator". Example; 'layer1_2_bn3 + layer1_1_relu_2'
            # NOTE: It will assumed that *ALWAYS* an operation is between spaces (expected: "3 + l"; unexpected: "3+l").
            #   Also it is asumed that there will be only one operator.
            operation = operation[0].split(OPERATION_SEPARATOR)
            op = operation.pop(1) # '0:layer1_2_bn3, 1:+, 2:layer1_1_relu_2            
            args = ''.join([LIST_START, LIST_SEPARATOR.join(operation), LIST_END]) #'[layer1_2_bn3, layer1_1_relu_2]' 
            # args now has the same format as other functions.
            func = cons.switch_operation_symbols(op)
    
        relations_dic[output_var] = (func, args)
    
    return relations_dic
# --- END extract_layers_relations --- #

def convert_layers_and_set_weights_and_biases(input_shape: Tuple[int], layers:Dict[str, Tuple[Union[str|torch.nn.Module], str]]) -> Tuple[List[LayerAndActivationBase], Dict[str, np.array], Dict[str, np.array]]:

    converted_layers: Dict[str, LayerAndActivationBase] = dict()

    # Constants 
    # - state_dicts keys. 
    LAYER_WEIGHTS = "weight"
    LAYER_BIASES = "bias"

    # - initalizers
    PYDTNN_WEIGHTS_INITIALIZER = "weights_initializer"
    PYDTNN_BIASES_INITIALIZER = "biases_initializer"
    # -----

    # NOTE: I didn't find a way to find the input shape from a PyTorch model ==> it became a parameter set by the user.
    layer_var_names = list(layers.keys())

    fst_layer = layer_var_names[0]
    _input = layers[fst_layer][1]
    converted_layers[_input] = ((Input(input_shape), None))

    dict_weights = dict()
    dict_biases = dict()
    # layer_var_names: {value's variable (str): ([function (str) or layer (nn.Module)], arguments (str))}
    for operation_variable in layer_var_names:
        operation, params = layers[operation_variable]

        if isinstance(operation, torch.nn.Module):
            layer = operation 
            layer_var = operation_variable 

            name = layer._get_name()
        
            args = {cons.ARGUMENTS: vars(layer)}
            # In this context, params are the input layers.
            converted_layer = cons.switch_pytorch_pydtnn(name)(args)           

            # -- Loading the weigths and the biases into the converted layer -- #
            # TODO: Check if there is another way to do this.
            def weights_initializer(*args_to_ignore):
                return dict_weights[operation]
            # - END weights_initializer - #

            # TODO: Check if there is another way to do this.
            def biases_initializer(*args_to_ignore):
                return dict_biases[operation]
            # - END weights_initializer - #

            state_dict = layer.state_dict()
                        
            # There are layers without weight nor biases
            if LAYER_WEIGHTS in state_dict:                        
                # The weights are "torch.Tensor": torch.Tensor.cpu().detach().numpy() ==> weigths as np.array
                dict_weights[operation] = state_dict[LAYER_WEIGHTS].cpu().detach().numpy()
                if hasattr(converted_layer, PYDTNN_WEIGHTS_INITIALIZER):
                    converted_layer.weights_initializer = weights_initializer
                else:
                    converted_layer.weights = dict_weights[operation]
            # else: Nothing special

            if LAYER_BIASES in state_dict: 
                dict_biases[operation] = state_dict[LAYER_BIASES].cpu().detach().numpy()                
                if hasattr(converted_layer, PYDTNN_BIASES_INITIALIZER):
                    converted_layer.biases_initializer = biases_initializer
                else:
                    converted_layer.biases = dict_biases[operation]
            # else: Nothing special

            # -- Loading the weigths and the biases into the converted layer -- #

            converted_layers[layer_var] = (converted_layer, params)

        else: #is intance of string (the name of a function or an operation)
            # Here, params are the input layers and other arguments.            
            args = {cons.PARAMETERS: params, cons.LAYERS: converted_layers}
            converted_layers[operation_variable] = cons.function_operation_to_pydtnn(operation)(args)
            # NOTE: Remember, originally theese were functions, then they does not have weights nor biases.
            # TODO: Check if it is necessary to set/unset in the class something (like the weigths update) in order to make it work like a function.
        
    list_layers = [layer for layer, _input in converted_layers.values()]
    return (list_layers, dict_weights, dict_biases)
# --- END convert_layers --- #

def convert_model(model:torch.nn.Module, input_shape:Tuple[int], omm=None, non_blocking_mpi=False, enable_gpu=False, enable_gpudirect=False,
                 enable_nccl=False, dtype=np.float32, tracing=False, tracer_output="", **kwargs) -> PyDTNN_Model:
    
    if "tensor_format" not in kwargs:
        kwargs["tensor_format"] = PYDTNN_TENSOR_FORMAT_NHWC #PYDTNN_TENSOR_FORMAT_NCHW #PYDTNN_TENSOR_FORMAT_NHWC
    if "model_name" not in kwargs:
        kwargs["model_name"] = None

    # Output model.
    converted_model = PyDTNN_Model(omm=omm, non_blocking_mpi=non_blocking_mpi, enable_gpu=enable_gpu, enable_gpudirect=enable_gpudirect,
                    enable_nccl=enable_nccl, dtype=dtype, tracing=tracing, tracer_output=tracer_output, **kwargs)    

    # Obtaining the model's layers/operations, activations, etc.; and the relation between them.
    dict_layers = extract_layers_relations(model = model)

    # Obtaining the PyDTNN equivalent
    layers, weights, biases = convert_layers_and_set_weights_and_biases(input_shape=input_shape, layers=dict_layers)

    # Asigning the layers/operations to the converted model.
    load_layers(model=converted_model, layers=layers)
        
    return converted_model
# --- END convert_model --- #      
