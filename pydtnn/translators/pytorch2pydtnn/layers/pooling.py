# _______________________________________________________________________________________________________________
# In this file must be implemented only the translation of PyTorch Pooling layers to its PyDTNN equivalent.
# _______________________________________________________________________________________________________________

# Typing related (or non important) imports
from typing import *
from inspect import stack # This is only in order to get the function's name

# Functionality imports
from pydtnn import layers
import pydtnn.translators.pytorch2pydtnn.common as cm

# ------------------- #
# ---- CONSTANTS ---- #
# ------------------- #

# PyTorch:
PYTORCH_KERNEL_SIZE = "kernel_size" # INT or Tuple[INT, INT]
PYTORCH_STRIDE = "stride" # INT or Tuple[INT, INT]
PYTORCH_PADDING = "padding" # INT or Tuple[INT, INT]
PYTORCH_DILATION = "dilation" # INT

# PyDTNN: 
PYDTNN_POOL_SHAPE = "pool_shape"
PYDTNN_STRIDE = "stride"
PYDTNN_PADDING = "padding"
PYDTNN_DILATION = "dilation"
# ------------------- #

# ------------------- #
def MaxPool2d(args: Dict[str, Any]) -> layers.MaxPool2D:
    # https://pytorch.org/docs/stable/generated/torch.nn.MaxPool2d.html#torch.nn.MaxPool2d

    print(f"Layer: {stack()[0].function}")
    # PyTorch attributes:
    # Not used: return_indices, ceil_mode
    torch_dict_keys = [PYTORCH_KERNEL_SIZE, PYTORCH_STRIDE, PYTORCH_PADDING, PYTORCH_DILATION]
    # ---- #

    # PyDTNN attributes:
    pydtnn_dict_keys = [PYDTNN_POOL_SHAPE, PYDTNN_STRIDE, PYDTNN_PADDING, PYDTNN_DILATION]
    # ---- #   

    layer_args = cm.prepare_pydtnn_arguments(arguments = args[cm.ARGUMENTS], torch_dict_keys = torch_dict_keys, pydtnn_dict_keys = pydtnn_dict_keys)

    if PYDTNN_POOL_SHAPE in layer_args:
        pool_shape = layer_args[PYDTNN_POOL_SHAPE]
        if isinstance(pool_shape, int):
            layer_args[PYDTNN_POOL_SHAPE] = (pool_shape, pool_shape)
        # else: It must be a Tuple[int, int], so it's okay
    # else: Nothing special
    
    return layers.MaxPool2D(**layer_args)
# --- END MaxPool2d --- #

def AvgPool2d(args: Dict[str, Any]) -> layers.AveragePool2D:    
    # https://pytorch.org/docs/stable/generated/torch.nn.AvgPool2d.html#torch.nn.AvgPool2d    
    
    print(f"Layer: {stack()[0].function}")
    # PyTorch attributes:
    # Not used: ceil_mode, count_include_pad, divisor_override
    torch_dict_keys = [PYTORCH_KERNEL_SIZE, PYTORCH_STRIDE, PYTORCH_PADDING, PYTORCH_DILATION]
    # ---- #

    # PyDTNN attributes:
    pydtnn_dict_keys = [PYDTNN_POOL_SHAPE, PYDTNN_STRIDE, PYDTNN_PADDING, PYDTNN_DILATION]
    # ---- #   

    layer_args = cm.prepare_pydtnn_arguments(arguments = args[cm.ARGUMENTS], torch_dict_keys = torch_dict_keys, pydtnn_dict_keys = pydtnn_dict_keys)

    if PYDTNN_POOL_SHAPE in layer_args:
        pool_shape = layer_args[PYDTNN_POOL_SHAPE]
        if isinstance(pool_shape, int):
            layer_args[PYDTNN_POOL_SHAPE] = (pool_shape, pool_shape)
        #else: It must be a Tuple[int, int], so it's okay

    return layers.AveragePool2D(**layer_args)
# --- END AvgPool2d --- #

def AdaptiveAvgPool2d(args: Dict[str, Any]) -> layers.AveragePool2D:
    # https://pytorch.org/docs/stable/generated/torch.nn.AdaptiveAvgPool2d.html#torch.nn.AdaptiveAvgPool2d
    # from torch.nn import AdaptiveAvgPool2d    
    # NOTE: "The output is of size H x W, for any input size. The number of output features is equal to the number of input planes." Source: PyTorch.      
    print(f"Layer: {stack()[0].function}")

    arguments = args[cm.ARGUMENTS]    
    output_shape = arguments[cm.PYTORCH_OUTPUT_SIZE] if cm.PYTORCH_OUTPUT_SIZE in arguments else None 

    return layers.AdaptiveAveragePool2D(output_shape=output_shape)
# --- END AdaptiveAvgPool2d --- #
