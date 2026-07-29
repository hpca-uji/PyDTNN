"""This module provides converters to translate PyTorch activation layers to their PyDTNN equivalents."""

import logging
from typing import Any

import pydtnn.converters.pytorch2pydtnn.common as cm
from pydtnn.activations.arctanh import Arctanh as Arctanh_PyDTNN
from pydtnn.activations.leaky_relu import LeakyRelu as LeakyRelu_PyDTNN
from pydtnn.activations.log_sigmoid import LogSigmoid as LogSigmoid_PyDTNN
from pydtnn.activations.log_softmax import LogSoftmax as LogSoftmax_PyDTNN
from pydtnn.activations.relu import Relu as Relu_PyDTNN
from pydtnn.activations.relu6 import Relu6 as Relu6_PyDTNN
from pydtnn.activations.sigmoid import Sigmoid as Sigmoid_PyDTNN
from pydtnn.activations.softmax import Softmax as Softmax_PyDTNN
from pydtnn.activations.tanh import Tanh as Tanh_PyDTNN

__all__ = (
    "Arctanh",
    "LeakyRelu",
    "LogSigmoid",
    "LogSoftmax",
    "ReLU",
    "ReLU6",
    "Sigmoid",
    "Softmax",
    "Tanh",
)

logger = logging.getLogger(__name__)

# Typing related (or non important) imports

# Functionality imports


def Arctanh(args: dict[str, Any]) -> Arctanh_PyDTNN:
    """
    Converts a PyTorch-like Arctanh_PyDTNN configuration to a PyDTNN Arctanh layer.

    Args:
        args: Dictionary containing layer configuration.

    Returns:
        An instance of pydtnn.activations.arctanh.Arctanh_PyDTNN.
    """
    # NOTE: There is no equivalent in PyTorch
    # not_used = args
    return Arctanh_PyDTNN()


def LogSigmoid(args: dict[str, Any]) -> LogSigmoid_PyDTNN:
    """
    Converts a PyTorch LogSigmoid layer to a PyDTNN Log layer.

    Args:
        args: Dictionary containing layer configuration.

    Returns:
        An instance of pydtnn.activations.log.Log_PyDTNN.
    """
    # https://pytorch.org/docs/stable/generated/torch.nn.LogSigmoid.html#torch.nn.LogSigmoid
    # not_used = args
    return LogSigmoid_PyDTNN()

def LogSoftmax(args: dict[str, Any]) -> LogSoftmax_PyDTNN:
    """
    Converts a PyTorch LogSoftmax layer to a PyDTNN Log layer.

    Args:
        args: Dictionary containing layer configuration.

    Returns:
        An instance of pydtnn.activations.log.Log_PyDTNN.
    """
    # https://pytorch.org/docs/stable/generated/torch.nn.LogSoftmax.html#torch.nn.LogSoftmax
    # not_used = args
    return LogSoftmax_PyDTNN()


def ReLU(args: dict[str, Any]) -> Relu_PyDTNN:
    """
    Converts a PyTorch ReLU layer to a PyDTNN Relu layer.

    Args:
        args: Dictionary containing layer configuration.

    Returns:
        An instance of pydtnn.activations.relu.Relu_PyDTNN.
    """
    # https://pytorch.org/docs/stable/generated/torch.nn.ReLU.html#torch.nn.ReLU
    # Not used Pytorch's parameters: inplace.
    # not_used = args
    return Relu_PyDTNN()


def ReLU6(args: dict[str, Any]) -> Relu6_PyDTNN:
    """
    Converts a PyTorch ReLU6 layer to a PyDTNN Relu6 layer.

    Args:
        args: Dictionary containing layer configuration.

    Returns:
        An instance of pydtnn.activations.relu6.Relu6_PyDTNN.
    """
    # https://pytorch.org/docs/stable/generated/torch.nn.ReLU.html#torch.nn.ReLU
    # Not used Pytorch's parameters: inplace.
    # not_used = args

    # NOTE: max_val. A interal PyTorch variable that seems to set the cap.

    return Relu6_PyDTNN()


def LeakyRelu(args: dict[str, Any]) -> LeakyRelu_PyDTNN:
    """
    Converts a PyTorch LeakyReLU layer to a PyDTNN LeakyRelu layer.

    Args:
        args: Dictionary containing layer configuration.

    Returns:
        An instance of pydtnn.activations.leaky_relu.LeakyRelu_PyDTNN.
    """
    # https://pytorch.org/docs/stable/generated/torch.nn.ReLU.html#torch.nn.ReLU
    # Not used Pytorch's parameters: inplace.
    negative_slope = "negative_slope"
    torch_dict_keys = [negative_slope]
    pydtnn_dict_keys = [negative_slope]

    layer_args = cm.prepare_pydtnn_arguments(
        arguments=args[cm.ARGUMENTS],
        torch_dict_keys=torch_dict_keys,
        pydtnn_dict_keys=pydtnn_dict_keys,
    )

    return LeakyRelu_PyDTNN(**layer_args)


def Sigmoid(args: dict[str, Any]) -> Sigmoid_PyDTNN:
    """
    Converts a PyTorch Sigmoid layer to a PyDTNN Sigmoid_PyDTNN layer.

    Args:
        args: Dictionary containing layer configuration.

    Returns:
        An instance of pydtnn.activations.sigmoid.Sigmoid_PyDTNN.
    """
    # https://pytorch.org/docs/stable/generated/torch.nn.Sigmoid_PyDTNN.html#torch.nn.Sigmoid_PyDTNN
    # not_used = args
    return Sigmoid_PyDTNN()


def Softmax(args: dict[str, Any]) -> Softmax_PyDTNN:
    """
    Converts a PyTorch Softmax layer to a PyDTNN Softmax_PyDTNN layer.

    Args:
        args: Dictionary containing layer configuration.

    Returns:
        An instance of pydtnn.activations.softmax.Softmax_PyDTNN.
    """
    # https://pytorch.org/docs/stable/generated/torch.nn.Softmax_PyDTNN.html#torch.nn.Softmax_PyDTNN
    # Not used Pytorch's parameters: dim.
    # not_used = args
    return Softmax_PyDTNN()


def Tanh(args: dict[str, Any]) -> Tanh_PyDTNN:
    """
    Converts a PyTorch Tanh layer to a PyDTNN Tanh_PyDTNN layer.

    Args:
        args: Dictionary containing layer configuration.

    Returns:
        An instance of pydtnn.activations.tanh.Tanh_PyDTNN.
    """
    # https://pytorch.org/docs/stable/generated/torch.nn.Tanh_PyDTNN.html#torch.nn.Tanh_PyDTNN
    # not_used = args
    return Tanh_PyDTNN()
