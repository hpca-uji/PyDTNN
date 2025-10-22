# _______________________________________________________________________________________________________________
# In this file must be implemented only the translation of PyTorch Activations layers to its PyDTNN equivalent.
# _______________________________________________________________________________________________________________

# Typing related (or non important) imports
from typing import *


# Functionality imports
from pydtnn import activations
import pydtnn.converters.pytorch2pydtnn.common as cm
# ------------------- #


def Arctanh(args: Dict[str, Any]) -> activations.Arctanh:
    # NOTE: There is no equivalent in PyTorch
    not_used = args
    return activations.Arctanh()


def LogSigmoid(args: Dict[str, Any]) -> activations.Log:
    # https://pytorch.org/docs/stable/generated/torch.nn.LogSigmoid.html#torch.nn.LogSigmoid
    not_used = args
    return activations.Log()


def ReLU(args: Dict[str, Any]) -> activations.Relu:
    # https://pytorch.org/docs/stable/generated/torch.nn.ReLU.html#torch.nn.ReLU
    # Not used Pytorch's parameters: inplace.
    not_used = args
    return activations.Relu()


def ReLU6(args: Dict[str, Any]) -> activations.Relu:
    # https://pytorch.org/docs/stable/generated/torch.nn.ReLU.html#torch.nn.ReLU
    # Not used Pytorch's parameters: inplace.
    not_used = args

    # NOTE: max_val. A interal PyTorch variable that seems to set the cap.

    return activations.Relu6()


def LeakyReLU(args: Dict[str, Any]) -> activations.Relu:
    # https://pytorch.org/docs/stable/generated/torch.nn.ReLU.html#torch.nn.ReLU
    # Not used Pytorch's parameters: inplace.
    NEGATIVE_SLOPE = "negative_slope"
    torch_dict_keys = [NEGATIVE_SLOPE]
    pydtnn_dict_keys = [NEGATIVE_SLOPE]

    layer_args = cm.prepare_pydtnn_arguments(arguments=args[cm.ARGUMENTS], torch_dict_keys=torch_dict_keys, pydtnn_dict_keys=pydtnn_dict_keys)

    return activations.LeakyRelu(**layer_args)


def Sigmoid(args: Dict[str, Any]) -> activations.Sigmoid:
    # https://pytorch.org/docs/stable/generated/torch.nn.Sigmoid.html#torch.nn.Sigmoid
    not_used = args
    return activations.Sigmoid()


def Softmax(args: Dict[str, Any]) -> activations.Softmax:
    # https://pytorch.org/docs/stable/generated/torch.nn.Softmax.html#torch.nn.Softmax
    # Not used Pytorch's parameters: dim.
    not_used = args
    return activations.Softmax()


def Tanh(args: Dict[str, Any]) -> activations.Tanh:
    # https://pytorch.org/docs/stable/generated/torch.nn.Tanh.html#torch.nn.Tanh
    not_used = args
    return activations.Tanh()
