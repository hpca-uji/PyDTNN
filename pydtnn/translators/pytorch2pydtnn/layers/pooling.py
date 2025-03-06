# _______________________________________________________________________________________________________________
# In this file must be implemented only the translation of PyTorch Pooling layers to its PyDTNN equivalent.
# _______________________________________________________________________________________________________________

# Typing related (or non important) imports
from typing import *
from inspect import stack # This is only in order to get the function's name

# Functionality imports
from pydtnn import layers
import pydtnn.translators.pytorch2pydtnn.constats as cons


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

    layer_args = cons.prepare_pydtnn_arguments(arguments = args[cons.ARGUMENTS], torch_dict_keys = torch_dict_keys, pydtnn_dict_keys = pydtnn_dict_keys)

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

    layer_args = cons.prepare_pydtnn_arguments(arguments = args[cons.ARGUMENTS], torch_dict_keys = torch_dict_keys, pydtnn_dict_keys = pydtnn_dict_keys)

    return layers.AveragePool2D(**layer_args)
# --- END AvgPool2d --- #

def AdaptiveAvgPool2d(args: Dict[str, Any]) -> layers.AveragePool2D:
    # https://pytorch.org/docs/stable/generated/torch.nn.AdaptiveAvgPool2d.html#torch.nn.AdaptiveAvgPool2d
    # from torch.nn import AdaptiveAvgPool2d    
    # NOTE: "The output is of size H x W, for any input size. The number of output features is equal to the number of input planes." Source: PyTorch.      
    
    arguments = args[cons.ARGUMENTS]
    _output_size = arguments[cons.PYTORCH_OUTPUT_SIZE] if cons.PYTORCH_OUTPUT_SIZE in arguments else None
    
    # TODO: Check if there is a way to avoid this (~terrorism~) *weird patch*.
    layer = layers.AveragePool2D()

    base_initialize = layer.initialize
    
    # NOTE: IMPORTANT if the base function's arguments change, it is necessary to change them here too.
    def new_initialize(self, prev_shape, need_dx=True):
        
        # https://stackoverflow.com/questions/58692476/what-is-adaptive-average-pooling-and-how-does-it-work 
        # https://stackoverflow.com/questions/53841509/how-does-adaptive-pooling-in-pytorch-work/63603993#63603993
        input_size = prev_shape
        output_size = _output_size if _output_size is not None else input_size
        layer.stride = input_size // output_size
        layer.pool_shape = input_size - (output_size-1) * layer.stride  
        layer.padding = 0
        
        # Using "layer.initialize" will produce a recursive call (Remember: this function will be "layer.initialize")
        base_initialize(self, prev_shape = prev_shape, need_dx = need_dx)
    # -- END new_initialize -- #
    
    layer.initialize = new_initialize

    return layer
# --- END AdaptiveAvgPool2d --- #