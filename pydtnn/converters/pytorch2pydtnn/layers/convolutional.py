# _______________________________________________________________________________________________________________
# In this file must be implemented only the translation of PyTorch Convolutional layers to its PyDTNN equivalent.
# _______________________________________________________________________________________________________________

# Typing related (or non important) imports
from typing import Dict, Any

# Functionality imports
from pydtnn.layers.conv_2d import Conv2D as _Conv2D
import pydtnn.converters.pytorch2pydtnn.common as cm

# ------------------ #


def Conv2d(args: Dict[str, Any]) -> _Conv2D:
    # https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html#torch.nn.Conv2d

    # PyTorch attributes:
    # Not used: in channels, device, dtype
    PYTORCH_KERNEL_SIZE = "kernel_size"  # INT or Tuple[INT, INT]
    PYTORCH_STRIDE = "stride"  # INT or Tuple[INT, INT]
    PYTORCH_PADDING = "padding"  # INT or Tuple[INT, INT]
    PYTORCH_DILATION = "dilation"  # INT
    PYTORCH_GROUPS = "groups"  # INT
    PYTORCH_BIAS = "bias"  # BOOL
    PYTORCH_OUPUT_CHANNELS = "out_channels"
    # PYTORCH_PADDING_MODE = "padding_mode" # STRING. Values: {"zeros", "reflect", "replicate", "circular"} | In PyDTNN "zeros" is the only implemented
    torch_dict_keys = [PYTORCH_KERNEL_SIZE, PYTORCH_STRIDE, PYTORCH_PADDING, PYTORCH_DILATION, PYTORCH_GROUPS, PYTORCH_BIAS, PYTORCH_OUPUT_CHANNELS]
    # ---- #

    # PyDTNN attributes:
    PYDTNN_FILTER_SHAPE = "filter_shape"
    PYDTNN_STRIDE = "stride"
    PYDTNN_PADDING = "padding"
    PYDTNN_DILATION = "dilation"
    PYDTNN_NFILTERS = "nfilters"
    PYDTNN_USE_BIAS = "use_bias"
    PYDTNN_NFILTERS = "nfilters"
    pydtnn_dict_keys = [PYDTNN_FILTER_SHAPE, PYDTNN_STRIDE, PYDTNN_PADDING, PYDTNN_DILATION, PYDTNN_NFILTERS, PYDTNN_USE_BIAS, PYDTNN_NFILTERS]
    # Not used: "activation" and "grouping"
    # Used, but in other place: "weights_initializer", "biases_initializer"
    # ---- #

    layer_args = cm.prepare_pydtnn_arguments(arguments=args[cm.ARGUMENTS], torch_dict_keys=torch_dict_keys, pydtnn_dict_keys=pydtnn_dict_keys)

    if PYDTNN_FILTER_SHAPE in layer_args:
        pool_shape = layer_args[PYDTNN_FILTER_SHAPE]
        if isinstance(pool_shape, int):
            layer_args[PYDTNN_FILTER_SHAPE] = (pool_shape, pool_shape)
        # else: It must be a Tuple[int, int], so it's okay
    # else: Nothing special

    return _Conv2D(**layer_args)
