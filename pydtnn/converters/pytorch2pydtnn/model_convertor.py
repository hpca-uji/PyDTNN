# Typing related
from typing import List, Dict, Tuple
from pydtnn.layer_base import LayerBase
from pydtnn.activations.activation import Activation
import numpy as np

# Operations/transformations related
import torch
from pydtnn.model import Model as PyDTNN_Model
from pydtnn.layers.input import Input
import pydtnn.converters.pytorch2pydtnn.common as cm
import copy


def load_layers(model: PyDTNN_Model, layers: List[LayerBase], activation_layer: Activation) -> None:
    for layer in layers:
        model.add(layer)
    if not isinstance(layers[-1], Activation) and activation_layer is not None:
        model.add(activation_layer)
    model._initialize()


def extract_layers_relations(model: torch.nn.Module) -> Dict[str, Tuple[str | torch.nn.Module, str]]:
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

    SEPARATOR_FUNCTION_VALUE = ";"
    SEPARATOR_ASSIGNATION = " = "
    PARAMETERS_BEGINING = "("
    PARAMETER_ENDING = ")"
    LIST_START = '['
    LIST_SEPARATOR = ','
    LIST_END = ']'
    OPERATION_SEPARATOR = " "  # It is expected that the operator is always between spaces (example: "a + b").

    MODEL_LAYER_REQ = "self"
    MODEL_FUNCT_ARG_NAME = "model"  # NOTE: "model" is the name of the function argument. If it change, it is necessary to change it here.

    TORCH_LAYER_REQ = "torch.nn.functional."
    TORCH_FUNC_REQ = "torch."
    PATTERNS = [TORCH_LAYER_REQ, TORCH_FUNC_REQ]  # NOTE: Order *IS* important.
    # -- END CONSTANTS -- #

    for line in filter(lambda x: not (FIRST_LINE in x or LAST_LINE in x),
                       filter(lambda x: len(x) != 0,
                              [elem.lstrip(PSEUDO_INDENTATION) for elem in graph.code.split(BY_LINES)])):

        # NOTE: seems that there are situations that the line does not have the value.
        line = line.split(SEPARATOR_FUNCTION_VALUE)[0]  # [line, debug's input's value]
        operation = line.split(SEPARATOR_ASSIGNATION)  # [output, function+args]
        if len(operation) > 2:
            # Case: When it is a call to a function with a keyword. Example: "cat = torch.concatenate([var], axis = 1)"
            output_var = operation.pop(0)
            operation = "=".join(operation)  # The spaces are removed to make easier a following step.
        else:
            # Normal case. Example: conv1 = self.conv1(x) ==> operation = [conv1, self.conv1(x)]
            output_var, operation = operation
        # Now we have split the _output's variable_ and the operation.
        # We want to separate the arguments from the function in order to get the layer and the relations with the previous layers.
        operation = operation.split(PARAMETERS_BEGINING)  # [function, ...n..., function, args)]

        func = None  # It will be assigned in the following if-else statement
        if len(operation) > 1:
            # Normal case. Examples: 'getattr(self.layer1, "2").bn1(layer1_2_conv1)', 'self.avgpool(features_36)'
            if any(MODEL_LAYER_REQ in part for part in operation):
                # Case: 'getattr(self.layer1, "2").bn1(layer1_2_conv1)'
                args = operation.pop().replace(PARAMETER_ENDING, "")  # [function, ...n..., function], args
                operation = PARAMETERS_BEGINING.join(operation)  # Reasembling the operation without the arguments.
                operation = operation.replace(MODEL_LAYER_REQ, MODEL_FUNCT_ARG_NAME)
                func = eval(operation)  # Getting the layer object.
            else:
                # Cases: function or layer not defined at model's object's constructor
                # TORCH_LAYER_REQ --> Case: layer not defined at model's object's constructor
                # Example: "adaptive_avg_pool2d = torch.nn.functional.adaptive_avg_pool2d(relu, (1, 1))" ==>
                # ==> operation = "torch.nn.functional.adaptive_avg_pool2d", args = "relu, (1, 1)"
                # NOTE: The first argument is always a previous layer
                # --
                # TORCH_FUNC_REQ --> Case: function. Example: torch.cat()
                _operation = operation.pop(0)  # _operation = something like "torch.cat"; operation= [arg1, arg2) arg3 etc.)] [list[str]]
                args = PARAMETERS_BEGINING.join(operation)[:-1]  # _operation = "torch.cat"; operation= arg1 (arg2) arg3 etc. [str] | [:-1] to remove the final ")"
                operation = _operation

                if operation in cm.SPECIAL_CASES:
                    # TODO [possible future FIXME]: See what to do with the special cases.
                    # continue
                    func = cm.CONCAT  # NOTE: this is a cheap fix. TODO: look what to do in this kind of situations.
                    # "torchvision_models_googlenet_GoogLeNetOutputs": The output is a tuple.
                for pattern in PATTERNS:
                    if pattern in operation:
                        func = operation.replace(pattern, "")  # operation = "adaptive_avg_pool2d" | "cat"
                        break
                    # else: Never happens. One (and only one) of the patterns *must* be in operation
        else:
            # Case "operator". Example; 'layer1_2_bn3 + layer1_1_relu_2'
            # NOTE: It will assumed that *ALWAYS* an operation is between spaces (expected: "3 + l"; unexpected: "3+l").
            #   Also it is asumed that there will be only one operator.
            operation = operation[0].split(OPERATION_SEPARATOR)
            op = operation.pop(1)  # '0:layer1_2_bn3, 1:+, 2:layer1_1_relu_2
            args = ''.join([LIST_START, LIST_SEPARATOR.join(operation), LIST_END])  # '[layer1_2_bn3, layer1_1_relu_2]'
            # args now has the same format as other functions.
            func = cm.switch_operation_symbols(op)
        relations_dic[output_var] = (func, args)
    # end "for line"

    return relations_dic


def convert_layers_and_set_weights_and_biases(input_shape: Tuple[int], layers: Dict[str, Tuple[str | torch.nn.Module, str]]) -> List[LayerBase]:

    converted_layers: Dict[str, LayerBase] = dict()

    # Constants
    # - state_dicts keys.
    LAYER_WEIGHTS = "weight"
    LAYER_BIASES = "bias"

    # - initalizers
    PYDTNN_WEIGHTS_INITIALIZER = "weights_initializer"
    PYDTNN_BIASES_INITIALIZER = "biases_initializer"
    # -
    # -----

    # NOTE: There is no way to get the input shape from a PyTorch model due depends of the dataset ==> The input shape will be a parameter set by the user.
    layer_var_names = list(layers.keys())

    fst_layer = layer_var_names[0]
    _input = layers[fst_layer][1]
    converted_layers[_input] = ((Input(input_shape), None))

    dict_equivalent_layer = dict()
    # If there are two layers like the following ones:
    #   "cat_1 = torch.cat([features_pool0, features_denseblock1_denselayer1_conv2], 1)"
    #   "cat_2 = torch.cat([features_pool0, features_denseblock1_denselayer1_conv2, features_denseblock1_denselayer2_conv2], 1)"
    # features_pool0, features_denseblock1_denselayer1_conv2 are actually "cat_1". The previous dictionary is used to make this "equivalence".

    # layer_var_names: {value's variable (str): ([function (str) or layer (nn.Module)], arguments (str))}
    for operation_variable in layer_var_names:
        operation, params = layers[operation_variable]

        if isinstance(operation, torch.nn.Module):
            layer = operation
            layer_var = operation_variable

            name = layer._get_name()

            # From "vars(layer)" it is possible to get all the information necessary from PyTorch in a dictionary.
            args = {cm.ARGUMENTS: vars(layer)}  # NOTE: In this context, params are the input layers.
            converted_layer = cm.switch_pytorch_pydtnn(name)(args)

            # -- Loading the weigths and the biases into the converted layer -- #
            state_dict = layer.state_dict()
            # There are layers without weight nor biases
            if LAYER_WEIGHTS in state_dict:
                # The weights are "torch.Tensor": torch.Tensor.cpu().detach().numpy() ==> weigths as np.array
                weights: np.ndarray = copy.deepcopy(state_dict[LAYER_WEIGHTS].cpu().detach().numpy())
                # NOTE: There are some layers (like the fully connected) where the shape in PyDTNN is the transpose of the PyTorch's one.
                weights = weights.T if name in cm.TRANSPOSE_WEIGHTS_LAYERS else weights

                if hasattr(converted_layer, PYDTNN_WEIGHTS_INITIALIZER):
                    def weights_initializer(shape: tuple, dtype: np.ndarray, pytorch_weights: np.ndarray = weights, **kwargs_to_ignore) -> np.ndarray:
                        # NOTE [IMPORTANT]: Regarding "pytorch_weights = weights".
                        # NOTE > If "weights" are directly set as the returned value (return weights), for some reason the return will be a reference to "weights"
                        # NOTE >> instead of the "weights" value (that is a reference to the layer's PyTorch's weights), so, since this is in a for loop and
                        # NOTE >>> this function (weights_initializer) is called in some step after the loop, every layer will have the last iteration's "weights" values
                        # NOTE >>>> -a reference to the last layer weights- instead of a reference to their respective layer weights.
                        # NOTE >>>>> In this way "pytorch_weights" has the copy of "weights" values (that, as said before, is a reference to the layer's weights) of that iteration.
                        return pytorch_weights.astype(dtype=dtype, copy=False)
                    # - END weights_initializer - #
                    setattr(converted_layer, PYDTNN_WEIGHTS_INITIALIZER, weights_initializer)
                else:
                    # I'm pretty sure this case never happens (anyways, it's better to have it just in case).
                    converted_layer.weights = weights
            # else: Nothing special.

            if LAYER_BIASES in state_dict:
                biases = copy.deepcopy(state_dict[LAYER_BIASES].cpu().detach().numpy())
                biases = biases.T if name in cm.TRANSPOSE_WEIGHTS_LAYERS else biases
                if hasattr(converted_layer, PYDTNN_BIASES_INITIALIZER):
                    def biases_initializer(shape: tuple, dtype: np.ndarray, pytorch_biases: np.ndarray = biases, **kwargs_to_ignore) -> np.ndarray:
                        # NOTE [IMPORTANT]: See "weights_initializer" notes; the case of "pytorch_biases = biases" parameter is the same case as weights_initializer's "pytorch_weights = weights".
                        return pytorch_biases.astype(dtype=dtype, copy=False)
                    # - END weights_initializer - #
                    setattr(converted_layer, PYDTNN_BIASES_INITIALIZER, biases_initializer)
                else:
                    # As said before, I'm pretty sure this case never happens, but anyways, it's better to have it just in case.
                    converted_layer.biases = biases
            # else: Nothing special.
            # -- END Loading the weigths and the biases into the converted layer -- #

            converted_layers[layer_var] = (converted_layer, params)
        else:  # is intance of string (the name of a function or an operation)
            # Here, params are the input layers and other arguments.
            args = {cm.PARAMETERS: params,
                    cm.LAYERS: converted_layers,
                    cm.EQUIVALENT_LAYERS: dict_equivalent_layer,
                    cm.OPERATION_VAR: operation_variable}

            converted_layers[operation_variable] = cm.function_operation_to_pydtnn(operation)(args)
            # NOTE: Remember, originally these were functions, then they does not have weights nor biases.
    # "for operation_variable in layer_var_names" end.

    list_layers = [layer for layer, _input in converted_layers.values()]
    return list_layers


def check_kwargs_and_set_default(kwargs: dict) -> None:

    DICT_KWARGS_DEFAULT_VALUES = {
        "tensor_format": "NCHW",  # NOTE: PyTorch's weight tensors only NCHW format.
        "model_name": None,  # NOTE: If it's not set to "None", it's possible that other neural network is loaded.
        "batch_size": 64,
        # Model object parameters:
        "omm": None,
        "enable_gpu": False,
        "enable_gpudirect": False,
        "non_blocking_mpi": False,
        "enable_nccl": False,
        "dtype": np.float32,
        "tracing": False,
        "tracer_output": "",
    }

    for k in DICT_KWARGS_DEFAULT_VALUES.keys():
        if k not in kwargs:
            kwargs[k] = DICT_KWARGS_DEFAULT_VALUES[k]


def convert_model(model: torch.nn.Module, input_shape: Tuple[int],
                  default_output_activation_layer: Activation | None = None, **kwargs) -> PyDTNN_Model:
    # "default_output_activation_layer" parameter: if there is no activation layer at the end, the one in this parameter is added to the converted model.
    check_kwargs_and_set_default(kwargs)

    # Output model.
    converted_model = PyDTNN_Model(**kwargs)

    # Obtaining the model's layers/operations, activations, etc.; and the relation between them.
    dict_layers = extract_layers_relations(model=model)

    # Obtaining the PyDTNN equivalent layer for every layer and setting the weights and biases (if it's necessary)
    layers = convert_layers_and_set_weights_and_biases(input_shape=input_shape, layers=dict_layers)

    # Assigning the layers/operations to the converted model and the default activation layer if there is none in the new model.
    load_layers(model=converted_model, layers=layers, activation_layer=default_output_activation_layer)

    return converted_model
