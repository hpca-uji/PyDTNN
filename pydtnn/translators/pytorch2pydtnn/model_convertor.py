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

    print("AAAAAAA") # TODO: BORRAR
    for line in filter(lambda x: not(FIRST_LINE in x or LAST_LINE in x) , 
                       filter(lambda x: len(x)!=0, 
                        [elem.lstrip(PSEUDO_INDENTATION) for elem in graph.code.split(BY_LINES)])):
        print(line) # TODO: BORRAR
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
            print(f"AAA: {operation}")            
            if any(MODEL_LAYER_REQ in part for part in operation):                
                # Case: 'getattr(self.layer1, "2").bn1(layer1_2_conv1)'
                args = operation.pop().replace(PARAMETER_ENDING, "") # [function, ...n..., function], args
                operation = PARAMETERS_BEGINING.join(operation) # Reasembling the operation without the arguments.
                print(f"operation: {operation}")
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
                print(f"operation: {operation}")
                _operation = operation.pop(0) # _operation = something like "torch.cat"; operation= [arg1, arg2) arg3 etc.)] [list[str]]
                args = PARAMETERS_BEGINING.join(operation)[:-1] # _operation = "torch.cat"; operation= arg1 (arg2) arg3 etc. [str] | [:-1] to remove the final ")"
                operation = _operation

                print(f"args: {args}")
                print(f"operation: {operation}")
                for pattern in [TORCH_LAYER_REQ, TORCH_FUNC_REQ]:
                    if pattern in operation:
                        print(f"pattern: {pattern}")
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

def convert_layers_and_get_weights_and_biases(layers:Dict[str, Tuple[Union[str|torch.nn.Module], str]]) -> Tuple[List[LayerAndActivationBase], Dict[str, np.array], Dict[str, np.array]]:

    converted_layers = dict()

    # Constants - state_dicts keys. 
    LAYER_WEIGHTS = "weight"
    LAYER_BIASES = "bias"
    # -----
    print(f"layers: {layers}")

    # TODO: Hacer bien.
    # TODO: Check if there is another way.
    # TODO: Handle situations where there are no input shape.
    layer_var_names = list(layers.keys())
    #first_layer_weights_bias = layers[layer_keys[0]].state_dict()
    #if LAYER_WEIGHTS in first_layer_weights_bias:
    #    first_layer_shape = layers[layer_keys[0]].state_dict()[LAYER_WEIGHTS].shape
    #else:
    #    first_layer_shape = (1,)   

    print(f"layer_var_names: {layer_var_names}")
    fst_layer = layer_var_names[0]
    print(f"fst_layer: {fst_layer}")
    _input = layers[fst_layer][1]
    # TODO: Extrat 1st layer in the proper way.
    first_layer_shape = (1,)

    converted_layers[_input] = ((Input(first_layer_shape), None))

    # TODO: Remove this:
    from pprint import pprint

    dict_weights = dict()
    dict_biases = dict()
    # layer_var_names: {value's variable (str): ([function (str) or layer (nn.Module)], arguments (str))}
    for operation_variable in layer_var_names:
        pprint(converted_layers, sort_dicts=False)
        operation, params = layers[operation_variable]
        print(f"operation: {operation}")
        print(f"params: {params}")

        # TODO:     
        # if ADD in name: 
        # TypeError: argument of type 'NoneType' is not iterable

        if isinstance(operation, torch.nn.Module):
                     
            layer = operation 
            layer_var = operation_variable 

            name = layer._get_name()
            state_dict = layer.state_dict()
            # There are layers without weight nor biases
            if LAYER_WEIGHTS in state_dict:
                # The weights are "torch.Tensor": torch.Tensor.cpu().detach().numpy() ==> weigths as np.array
                dict_weights[name] = state_dict[LAYER_WEIGHTS].cpu().detach().numpy()
            if LAYER_BIASES in state_dict: 
                dict_biases[name] = state_dict[LAYER_BIASES].cpu().detach().numpy()
        
            args = {cons.ARGUMENTS: vars(layer)}
            # In this context, params are the input layers.
            converted_layers[layer_var] = (cons.switch_pytorch_pydtnn(name)(args), params)

        else: #is intance of string (the name of a function or an operation)
            # Here, params are the input layers and other arguments.            
            args = {cons.PARAMETERS: params, cons.LAYERS: converted_layers}
            converted_layers[operation_variable] = cons.function_operation_to_pydtnn(operation)(args)
        
    return (converted_layers, dict_weights, dict_biases)
# --- END convert_layers --- #

def convert_model(model:torch.nn.Module, omm=None, non_blocking_mpi=False, enable_gpu=False, enable_gpudirect=False,
                 enable_nccl=False, dtype=np.float32, tracing=False, tracer_output="", **kwargs) -> PyDTNN_Model:
    
    if "tensor_format" not in kwargs:
        kwargs["tensor_format"] = PYDTNN_TENSOR_FORMAT_NHWC # PYDTNN_TENSOR_FORMAT_NCHW #PYDTNN_TENSOR_FORMAT_NHWC
    if "model_name" not in kwargs:
        kwargs["model_name"] = None

    # Output model.
    # NOTE: ¡¡¡¡IMPORTANT!!!!! Be sure that the "parser.model_name" from "pydtnn.parser" is None!!!!!!!!.
    converted_model = PyDTNN_Model(omm=omm, non_blocking_mpi=non_blocking_mpi, enable_gpu=enable_gpu, enable_gpudirect=enable_gpudirect,
                    enable_nccl=enable_nccl, dtype=dtype, tracing=tracing, tracer_output=tracer_output, **kwargs)    

    # Obtaining the model's layers/operations, activations, etc.; and the relation between them.
    dict_layers = extract_layers_relations(model = model)

    # Obtaining the PyDTNN equivalent
    layers, weights, biases = convert_layers_and_get_weights_and_biases(dict_layers)

    # Asigning the layers/operations to the converted model.
    load_layers(model=converted_model, layers=layers)

    # Loading the weights into the model.
    converted_model.load_store_path(layers = converted_model.layers, d = weights, mode = "load")
        
    return converted_model
# --- END convert_model --- #      
