# _______________________________________________________________________________________________________________
# In this file must be implemented only the translation of PyTorch Convolutional layers to its PyDTNN equivalent.
# _______________________________________________________________________________________________________________

# Typing related (or non important) imports
from typing import *
from inspect import stack # This is only in order to get the function's name

# Functionality imports
from pydtnn import layers
import pydtnn.translators.pytorch2pydtnn.constats as cons

# ------------------ #
def Conv2d(args: Dict[str, Any]) -> layers.Conv2D:    
    # https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html#torch.nn.Conv2d

    print(f"Layer: {stack()[0].function}")
    # PyTorch attributes:
    # Not used: in/out channels, device, dtype
    PYTORCH_KERNEL_SIZE = "kernel_size" # INT or Tuple[INT, INT]
    PYTORCH_STRIDE = "stride" # INT or Tuple[INT, INT]
    PYTORCH_PADDING = "padding" # INT or Tuple[INT, INT]
    PYTORCH_DILATION = "dilation" # INT
    PYTORCH_GROUPS = "groups" # INT
    PYTORCH_BIAS = "bias" # BOOL
    # PYTORCH_PADDING_MODE = "padding_mode" # STRING. Values: {"zeros", "reflect", "replicate", "circular"} | In PyDTNN only implemented Zeros
    torch_dict_keys = [PYTORCH_KERNEL_SIZE, PYTORCH_STRIDE, PYTORCH_PADDING, PYTORCH_DILATION, PYTORCH_GROUPS, PYTORCH_BIAS]
    # ---- #

    # PyDTNN attributes:
    PYDTNN_FILTER_SHAPE = "filter_shape"
    PYDTNN_STRIDE = "stride"
    PYDTNN_PADDING = "padding"
    PYDTNN_DILATION = "dilation"
    PYDTNN_NFILTERS = "nfilters"
    PYDTNN_USE_BIAS = "use_bias"
    pydtnn_dict_keys = [PYDTNN_FILTER_SHAPE, PYDTNN_STRIDE, PYDTNN_PADDING, PYDTNN_DILATION, PYDTNN_NFILTERS, PYDTNN_USE_BIAS]
    # Not used: "grouping" "activation" "weights_initializer" "biases_initializer"
    # ---- #   

    layer_args = cons.prepare_pydtnn_arguments(arguments = args[cons.ARGUMENTS], torch_dict_keys = torch_dict_keys, pydtnn_dict_keys = pydtnn_dict_keys)

    return layers.Conv2D(**layer_args)
# --- END Conv2d --- #