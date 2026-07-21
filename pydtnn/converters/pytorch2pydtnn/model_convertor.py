"""Module for converting PyTorch models to PyDTNN models."""

import copy
import logging
from typing import Any

import numpy as np
import torch  # type: ignore

import pydtnn.converters.pytorch2pydtnn.common as cm
from pydtnn.abstract.layerable import Layerable
from pydtnn.activations.abstract.activation import Activation
from pydtnn.layers.batch_normalization import BatchNormalization
from pydtnn.layers.input import Input
from pydtnn.model import Model as PyDTNN_Model
from pydtnn.utils.constants import ArrayShape
from pydtnn.utils.tensor import TensorFormat

__all__ = (
    "check_kwargs_and_set_default",
    "convert_layers",
    "convert_model",
    "extract_layers_relations",
    "load_layers",
)

logger = logging.getLogger(__name__)

# Typing related

# Operations/transformations related


def load_layers(model: PyDTNN_Model, layers: list[Layerable]) -> None:
    """
    Adds a list of layers to the model and initializes it.

    Args:
        model: The PyDTNN model to populate.
        layers: A list of layer objects to add.
        activation_layer: An optional activation layer to append if not present.
    """
    for layer in layers:
        model.add(layer)
    model._model_init()


def extract_layers_relations(
    model: torch.nn.Module,
) -> dict[str, tuple[str | torch.nn.Module, str]]:
    """
    Parses a PyTorch model to extract layer relationships and operations.

    Args:
        model: The PyTorch model to trace.

    Returns:
        A dictionary mapping output variable names to a tuple of (operation/layer, arguments).
    """
    # TODO: Search the way "torch.fx.symbolic_trace" generates ".code" and not extracting the data from a
    # graph: torch.fx.GraphModule
    graph = torch.fx.symbolic_trace(model)

    # {[output's variable name]: tuple([string with operation name or the layer object], [string with the args])}
    relations_dic = dict()

    # -- CONSTANTS --
    by_lines = "\n"
    pseudo_indentation = " "
    first_line = "forward"
    last_line = "return"

    separator_function_value = ";"
    separator_assignation = " = "
    parameters_begining = "("
    parameter_ending = ")"
    list_start = "["
    list_separator = ","
    list_end = "]"
    operation_separator = (  # It is expected that the operator is always between spaces (example: "a + b").
        " "
    )

    model_layer_req = "self"
    # NOTE: "model" is the name of the function argument. If it change, it is
    # necessary to change it here.
    model_funct_arg_name = "model"

    torch_layer_req = "torch.nn.functional."
    torch_func_req = "torch."
    patterns = [torch_layer_req, torch_func_req]  # NOTE: Order *IS* important.

    for line in filter(
        lambda x: not (first_line in x or last_line in x),
        filter(
            lambda x: len(x) != 0,
            [elem.lstrip(pseudo_indentation) for elem in graph.code.split(by_lines)],
        ),
    ):
        # NOTE: seems that there are situations that the line does not have the value.
        line = line.split(separator_function_value)[0]  # [line, debug's input's value]
        operation = line.split(separator_assignation)  # [output, function+args]
        if len(operation) > 2:
            # Case: When it is a call to a function with a keyword. Example: "cat =
            # torch.concatenate([var], axis = 1)"
            output_var = operation.pop(0)
            operation = "=".join(
                operation
            )  # The spaces are removed to make easier a following step.
        else:
            # Normal case. Example: conv1 = self.conv1(x) ==> operation = [conv1, self.conv1(x)]
            output_var, operation = operation
        # Now we have split the _output's variable_ and the operation.
        # We want to separate the arguments from the function in order to get the
        # layer and the relations with the previous layers.
        operation = operation.split(parameters_begining)  # [function, ...n..., function, args)]

        func = None  # It will be assigned in the following if-else statement
        if len(operation) > 1:
            # Normal case. Examples: 'getattr(self.layer1, "2").bn1(layer1_2_conv1)',
            # 'self.avgpool(features_36)'
            if any(model_layer_req in part for part in operation):
                # Case: 'getattr(self.layer1, "2").bn1(layer1_2_conv1)'
                args = operation.pop().replace(
                    parameter_ending, ""
                )  # [function, ...n..., function], args
                operation = parameters_begining.join(
                    operation
                )  # Reasembling the operation without the arguments.
                operation = operation.replace(model_layer_req, model_funct_arg_name)
                func = eval(operation)  # Getting the layer object.
            else:
                # Cases: function or layer not defined at model's object's constructor
                # TORCH_LAYER_REQ --> Case: layer not defined at model's object's constructor
                # Example: "adaptive_avg_pool2d = torch.nn.functional.adaptive_avg_pool2d(relu, (1, 1))" ==>
                # NOTE: The first argument is always a previous layer
                # TORCH_FUNC_REQ --> Case: function. Example: torch.cat()
                # _operation = something like "torch.cat"; operation= [arg1, arg2) arg3
                # etc.)] [list[str]]
                _operation = operation.pop(0)
                # _operation = "torch.cat"; operation= arg1 (arg2) arg3 etc. [str] | [:-1]
                # to remove the final ")"
                args = parameters_begining.join(operation)[:-1]
                operation = _operation

                if operation in cm.SPECIAL_CASES:
                    # TODO [possible future FIXME]: See what to do with the special cases.
                    # continue
                    # NOTE: this is a cheap fix. TODO: look what to do in this kind of situations.
                    func = cm.CONCAT
                    # "torchvision_models_googlenet_GoogLeNetOutputs": The output is a tuple.
                for pattern in patterns:
                    if pattern in operation:
                        func = operation.replace(
                            pattern, ""
                        )  # operation = "adaptive_avg_pool2d" | "cat"
                        break
                    # else: Never happens. One (and only one) of the patterns *must* be in operation
        else:
            # Case "operator". Example; 'layer1_2_bn3 + layer1_1_relu_2'
            # NOTE: It will assumed that *ALWAYS* an operation is between spaces (expected: "3 + l"; unexpected: "3+l").
            #   Also it is asumed that there will be only one operator.
            operation = operation[0].split(operation_separator)
            op = operation.pop(1)  # '0:layer1_2_bn3, 1:+, 2:layer1_1_relu_2
            args = "".join(
                [list_start, list_separator.join(operation), list_end]
            )  # '[layer1_2_bn3, layer1_1_relu_2]'
            # args now has the same format as other functions.
            func = cm.switch_operation_symbols(op)
        relations_dic[output_var] = (func, args)
    # end "for line"

    return relations_dic


def convert_layers(
    input_shape: ArrayShape, layers: dict[str, tuple[str | torch.nn.Module, str]]
) -> list[Layerable]:
    """
    Converts PyTorch layers to PyDTNN layers and maps weights/biases.

    Args:
        input_shape: The input shape of the model.
        layers: Dictionary of layer relations extracted from the model.

    Returns:
        A list of converted PyDTNN layers.
    """

    converted_layers: dict[str, tuple[Layerable, str | None]] = dict()

    # NOTE: There is no way to get the input shape from a PyTorch model due
    # depends of the dataset ==> The input shape will be a parameter set by
    # the user.
    layer_var_names = list(layers.keys())

    fst_layer = layer_var_names[0]
    _input = layers[fst_layer][1]
    converted_layers[_input] = (Input(input_shape), None)

    dict_equivalent_layer = dict()
    # If there are two layers like the following ones:
    #   "cat_1 = torch.cat([features_pool0, features_denseblock1_denselayer1_conv2], 1)"
    #   "cat_2 = torch.cat([features_pool0, features_denseblock1_denselayer1_conv2, features_denseblock1_denselayer2_conv2], 1)"
    # features_pool0, features_denseblock1_denselayer1_conv2 are actually
    # "cat_1". The previous dictionary is used to make this "equivalence".

    # layer_var_names: {value's variable (str): ([function (str) or layer
    # (nn.Module)], arguments (str))}
    for operation_variable in layer_var_names:
        operation, params = layers[operation_variable]

        if isinstance(operation, torch.nn.Module):
            layer: torch.nn.Module = operation
            layer_var = operation_variable

            name = layer._get_name()

            # From "vars(layer)" it is possible to get all the information necessary
            # from PyTorch in a dictionary.
            args = {
                cm.ARGUMENTS: vars(layer)
            }  # NOTE: In this context, params are the input layers.
            converted_layer = cm.switch_pytorch_pydtnn(name)(args)

            converted_layers[layer_var] = (converted_layer, params)
        else:  # is intance of string (the name of a function or an operation)
            # Here, params are the input layers and other arguments.
            args = {
                cm.PARAMETERS: params,
                cm.LAYERS: converted_layers,
                cm.EQUIVALENT_LAYERS: dict_equivalent_layer,
                cm.OPERATION_VAR: operation_variable,
            }

            converted_layers[operation_variable] = cm.function_operation_to_pydtnn(operation)(args)
            # NOTE: Remember, originally these were functions, then they does not have weights nor biases.
    # "for operation_variable in layer_var_names" end.

    list_layers = [layer for layer, _input in converted_layers.values()]
    return list_layers


def check_kwargs_and_set_default(kwargs: dict) -> None:
    """
    Sets default values for missing keyword arguments.

    Args:
        kwargs: Dictionary of user-provided arguments.
    """

    assert kwargs.get("tensor_format") == TensorFormat.NCHW, (
        "PyTorch is only implemented for NCHW format"
    )
    kwargs["model_name"] = None


def get_layers_from_torch(
    model: torch.nn.Module,
    input_shape: ArrayShape,
    default_output_activation_layer: Activation | None = None,
) -> list[Layerable]:
    """
    Get a list of the equivalent PyDTNN's layers from a PyTorch model.

    Args:
        model: The PyTorch model to convert.
        input_shape: The input shape of the model.
        default_output_activation_layer: Optional activation layer to add at the end.

    Returns:
        A list with the equivalent PyDTNN's layers from a PyTorch model.
    """
    # Obtaining the model's layers/operations, activations, etc.; and the relation between them.
    dict_layers = extract_layers_relations(model=model)

    # Obtaining the PyDTNN equivalent layer for every layer and setting the
    # weights and biases (if it's necessary)
    layers = convert_layers(input_shape=input_shape, layers=dict_layers)

    if not isinstance(layers[-1], Activation) and default_output_activation_layer is not None:
        layers.append(default_output_activation_layer)

    return layers


def convert_model(
    model: torch.nn.Module,
    input_shape: ArrayShape,
    default_output_activation_layer: Activation | None = None,
    **kwargs: Any,
) -> PyDTNN_Model:
    """
    Converts a PyTorch model to a PyDTNN model.

    Args:
        model: The PyTorch model to convert.
        input_shape: The input shape of the model.
        default_output_activation_layer: Optional activation layer to add at the end.
        **kwargs: Additional configuration parameters.

    Returns:
        A converted PyDTNN model.
    """
    # "default_output_activation_layer" parameter: if there is no activation layer at the end,
    #  the one in this parameter is added to the converted model.
    check_kwargs_and_set_default(kwargs)

    # Output model.
    converted_model = PyDTNN_Model(**kwargs)

    layers = get_layers_from_torch(model, input_shape, default_output_activation_layer)

    # Assigning the layers/operations to the converted model and the default
    # activation layer if there is none in the new model.
    load_layers(model=converted_model, layers=layers)

    return converted_model
