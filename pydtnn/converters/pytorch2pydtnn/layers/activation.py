# _______________________________________________________________________________________________________________
# In this file must be implemented only the translation of PyTorch Activations layers to its PyDTNN equivalent.
# _______________________________________________________________________________________________________________

# Typing related (or non important) imports
from typing import Dict, Any


# Functionality imports
from pydtnn.activations.arctanh import Arctanh as _Arctanh
from pydtnn.activations.log import Log as _Log
from pydtnn.activations.relu import Relu as _Relu
from pydtnn.activations.relu6 import Relu6 as _Relu6
from pydtnn.activations.leaky_relu import LeakyRelu as _LeakyRelu
from pydtnn.activations.sigmoid import Sigmoid as _Sigmoid
from pydtnn.activations.softmax import Softmax as _Softmax
from pydtnn.activations.tanh import Tanh as _Tanh
import pydtnn.converters.pytorch2pydtnn.common as cm
# ------------------- #


def Arctanh(args: Dict[str, Any]) -> _Arctanh:
    # NOTE: There is no equivalent in PyTorch
    not_used = args
    return _Arctanh()


def LogSigmoid(args: Dict[str, Any]) -> _Log:
    # https://pytorch.org/docs/stable/generated/torch.nn.LogSigmoid.html#torch.nn.LogSigmoid
    not_used = args
    return _Log()


def ReLU(args: Dict[str, Any]) -> _Relu:
    # https://pytorch.org/docs/stable/generated/torch.nn.ReLU.html#torch.nn.ReLU
    # Not used Pytorch's parameters: inplace.
    not_used = args
    return _Relu()


def ReLU6(args: Dict[str, Any]) -> _Relu:
    # https://pytorch.org/docs/stable/generated/torch.nn.ReLU.html#torch.nn.ReLU
    # Not used Pytorch's parameters: inplace.
    not_used = args

    # NOTE: max_val. A interal PyTorch variable that seems to set the cap.

    return _Relu6()


def LeakyReLU(args: Dict[str, Any]) -> _Relu:
    # https://pytorch.org/docs/stable/generated/torch.nn.ReLU.html#torch.nn.ReLU
    # Not used Pytorch's parameters: inplace.
    NEGATIVE_SLOPE = "negative_slope"
    torch_dict_keys = [NEGATIVE_SLOPE]
    pydtnn_dict_keys = [NEGATIVE_SLOPE]

    layer_args = cm.prepare_pydtnn_arguments(arguments=args[cm.ARGUMENTS], torch_dict_keys=torch_dict_keys, pydtnn_dict_keys=pydtnn_dict_keys)

    return _LeakyRelu(**layer_args)


def Sigmoid(args: Dict[str, Any]) -> _Sigmoid:
    # https://pytorch.org/docs/stable/generated/torch.nn.Sigmoid.html#torch.nn.Sigmoid
    not_used = args
    return _Sigmoid()


def Softmax(args: Dict[str, Any]) -> _Softmax:
    # https://pytorch.org/docs/stable/generated/torch.nn.Softmax.html#torch.nn.Softmax
    # Not used Pytorch's parameters: dim.
    not_used = args
    return _Softmax()


def Tanh(args: Dict[str, Any]) -> _Tanh:
    # https://pytorch.org/docs/stable/generated/torch.nn.Tanh.html#torch.nn.Tanh
    not_used = args
    return _Tanh()
