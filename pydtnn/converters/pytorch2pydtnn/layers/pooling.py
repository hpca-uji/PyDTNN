# _______________________________________________________________________________________________________________
# In this file must be implemented only the translation of PyTorch Pooling layers to its PyDTNN equivalent.
# _______________________________________________________________________________________________________________

# Typing related (or non important) imports
from typing import Dict, Any

# Functionality imports
from pydtnn.layers.max_pool_2d import MaxPool2D
from pydtnn.layers.average_pool_2d import AveragePool2D
from pydtnn.layers.adaptive_average_pool_2d import AdaptiveAveragePool2D
import pydtnn.converters.pytorch2pydtnn.common as cm

# ------------------- #
# ---- CONSTANTS ---- #
# ------------------- #

# PyTorch:
PYTORCH_KERNEL_SIZE = "kernel_size"  # INT or Tuple[INT, INT]
PYTORCH_STRIDE = "stride"  # INT or Tuple[INT, INT]
PYTORCH_PADDING = "padding"  # INT or Tuple[INT, INT]
PYTORCH_DILATION = "dilation"  # INT

# PyDTNN:
PYDTNN_POOL_SHAPE = "pool_shape"
PYDTNN_STRIDE = "stride"
PYDTNN_PADDING = "padding"
PYDTNN_DILATION = "dilation"
# ------------------- #

# ------------------- #


def MaxPool2d(args: Dict[str, Any]) -> MaxPool2D:
    # https://pytorch.org/docs/stable/generated/torch.nn.MaxPool2d.html#torch.nn.MaxPool2d

    # PyTorch attributes:
    # Not used: return_indices, ceil_mode
    torch_dict_keys = [PYTORCH_KERNEL_SIZE, PYTORCH_STRIDE, PYTORCH_PADDING, PYTORCH_DILATION]
    # ---- #

    # PyDTNN attributes:
    pydtnn_dict_keys = [PYDTNN_POOL_SHAPE, PYDTNN_STRIDE, PYDTNN_PADDING, PYDTNN_DILATION]
    # ---- #

    layer_args = cm.prepare_pydtnn_arguments(arguments=args[cm.ARGUMENTS], torch_dict_keys=torch_dict_keys, pydtnn_dict_keys=pydtnn_dict_keys)

    if PYDTNN_POOL_SHAPE in layer_args:
        pool_shape = layer_args[PYDTNN_POOL_SHAPE]
        if isinstance(pool_shape, int):
            layer_args[PYDTNN_POOL_SHAPE] = (pool_shape, pool_shape)
        # else: It must be a Tuple[int, int], so it's okay
    # else: Nothing special

    return MaxPool2D(**layer_args)


def AvgPool2d(args: Dict[str, Any]) -> AveragePool2D:
    # https://pytorch.org/docs/stable/generated/torch.nn.AvgPool2d.html#torch.nn.AvgPool2d

    # PyTorch attributes:
    # Not used: ceil_mode, count_include_pad, divisor_override
    torch_dict_keys = [PYTORCH_KERNEL_SIZE, PYTORCH_STRIDE, PYTORCH_PADDING, PYTORCH_DILATION]
    # ---- #

    # PyDTNN attributes:
    pydtnn_dict_keys = [PYDTNN_POOL_SHAPE, PYDTNN_STRIDE, PYDTNN_PADDING, PYDTNN_DILATION]
    # ---- #

    layer_args = cm.prepare_pydtnn_arguments(arguments=args[cm.ARGUMENTS], torch_dict_keys=torch_dict_keys, pydtnn_dict_keys=pydtnn_dict_keys)

    if PYDTNN_POOL_SHAPE in layer_args:
        pool_shape = layer_args[PYDTNN_POOL_SHAPE]
        if isinstance(pool_shape, int):
            layer_args[PYDTNN_POOL_SHAPE] = (pool_shape, pool_shape)
        # else: It must be a Tuple[int, int], so it's okay

    return AveragePool2D(**layer_args)


def AdaptiveAvgPool2d(args: Dict[str, Any]) -> AdaptiveAveragePool2D:
    # https://pytorch.org/docs/stable/generated/torch.nn.AdaptiveAvgPool2d.html#torch.nn.AdaptiveAvgPool2d
    # from torch.nn import AdaptiveAvgPool2d

    arguments = args[cm.ARGUMENTS]
    output_shape = arguments[cm.PYTORCH_OUTPUT_SIZE] if cm.PYTORCH_OUTPUT_SIZE in arguments else None

    return AdaptiveAveragePool2D(output_shape=output_shape)
