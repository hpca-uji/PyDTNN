"""
This module provides converters to translate PyTorch activation layers to their PyDTNN equivalents.
"""

import logging
from typing import Any

import pydtnn.converters.pytorch2pydtnn.common as cm
from pydtnn.activations.arctanh import Arctanh as _Arctanh
from pydtnn.activations.leaky_relu import LeakyRelu as _LeakyRelu
from pydtnn.activations.log import Log as _Log
from pydtnn.activations.relu import Relu as _Relu
from pydtnn.activations.relu6 import Relu6 as _Relu6
from pydtnn.activations.sigmoid import Sigmoid as _Sigmoid
from pydtnn.activations.softmax import Softmax as _Softmax
from pydtnn.activations.tanh import Tanh as _Tanh

__all__ = (
    "Arctanh",
    "LeakyReLU",
    "LogSigmoid",
    "ReLU",
    "ReLU6",
    "Sigmoid",
    "Softmax",
    "Tanh",
)

logger = logging.getLogger(__name__)

# Typing related (or non important) imports

# Functionality imports


def Arctanh(args: dict[str, Any]) -> _Arctanh:
    """
    Converts a PyTorch-like Arctanh configuration to a PyDTNN Arctanh layer.

    Args:
        args: Dictionary containing layer configuration.

    Returns:
        An instance of pydtnn.activations.arctanh.Arctanh.
    """
    # NOTE: There is no equivalent in PyTorch
    # not_used = args
    return _Arctanh()


def LogSigmoid(args: dict[str, Any]) -> _Log:
    """
    Converts a PyTorch LogSigmoid layer to a PyDTNN Log layer.

    Args:
        args: Dictionary containing layer configuration.

    Returns:
        An instance of pydtnn.activations.log.Log.
    """
    # https://pytorch.org/docs/stable/generated/torch.nn.LogSigmoid.html#torch.nn.LogSigmoid
    # not_used = args
    return _Log()


def ReLU(args: dict[str, Any]) -> _Relu:
    """
    Converts a PyTorch ReLU layer to a PyDTNN Relu layer.

    Args:
        args: Dictionary containing layer configuration.

    Returns:
        An instance of pydtnn.activations.relu.Relu.
    """
    # https://pytorch.org/docs/stable/generated/torch.nn.ReLU.html#torch.nn.ReLU
    # Not used Pytorch's parameters: inplace.
    # not_used = args
    return _Relu()


def ReLU6(args: dict[str, Any]) -> _Relu:
    """
    Converts a PyTorch ReLU6 layer to a PyDTNN Relu6 layer.

    Args:
        args: Dictionary containing layer configuration.

    Returns:
        An instance of pydtnn.activations.relu6.Relu6.
    """
    # https://pytorch.org/docs/stable/generated/torch.nn.ReLU.html#torch.nn.ReLU
    # Not used Pytorch's parameters: inplace.
    # not_used = args

    # NOTE: max_val. A interal PyTorch variable that seems to set the cap.

    return _Relu6()


def LeakyReLU(args: dict[str, Any]) -> _Relu:
    """
    Converts a PyTorch LeakyReLU layer to a PyDTNN LeakyRelu layer.

    Args:
        args: Dictionary containing layer configuration.

    Returns:
        An instance of pydtnn.activations.leaky_relu.LeakyRelu.
    """
    # https://pytorch.org/docs/stable/generated/torch.nn.ReLU.html#torch.nn.ReLU
    # Not used Pytorch's parameters: inplace.
    NEGATIVE_SLOPE = "negative_slope"
    torch_dict_keys = [NEGATIVE_SLOPE]
    pydtnn_dict_keys = [NEGATIVE_SLOPE]

    layer_args = cm.prepare_pydtnn_arguments(
        arguments=args[cm.ARGUMENTS],
        torch_dict_keys=torch_dict_keys,
        pydtnn_dict_keys=pydtnn_dict_keys,
    )

    return _LeakyRelu(**layer_args)


def Sigmoid(args: dict[str, Any]) -> _Sigmoid:
    """
    Converts a PyTorch Sigmoid layer to a PyDTNN Sigmoid layer.

    Args:
        args: Dictionary containing layer configuration.

    Returns:
        An instance of pydtnn.activations.sigmoid.Sigmoid.
    """
    # https://pytorch.org/docs/stable/generated/torch.nn.Sigmoid.html#torch.nn.Sigmoid
    # not_used = args
    return _Sigmoid()


def Softmax(args: dict[str, Any]) -> _Softmax:
    """
    Converts a PyTorch Softmax layer to a PyDTNN Softmax layer.

    Args:
        args: Dictionary containing layer configuration.

    Returns:
        An instance of pydtnn.activations.softmax.Softmax.
    """
    # https://pytorch.org/docs/stable/generated/torch.nn.Softmax.html#torch.nn.Softmax
    # Not used Pytorch's parameters: dim.
    # not_used = args
    return _Softmax()


def Tanh(args: dict[str, Any]) -> _Tanh:
    """
    Converts a PyTorch Tanh layer to a PyDTNN Tanh layer.

    Args:
        args: Dictionary containing layer configuration.

    Returns:
        An instance of pydtnn.activations.tanh.Tanh.
    """
    # https://pytorch.org/docs/stable/generated/torch.nn.Tanh.html#torch.nn.Tanh
    # not_used = args
    return _Tanh()
