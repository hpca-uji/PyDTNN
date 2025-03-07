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

    layer_args = cons.prepare_pydtnn_arguments(arguments = args[cons.ARGUMENTS], torch_dict_keys = torch_dict_keys, pydtnn_dict_keys = pydtnn_dict_keys)

    if PYDTNN_POOL_SHAPE in layer_args:
        pool_shape = layer_args[PYDTNN_POOL_SHAPE]
        if isinstance(pool_shape, int):
            layer_args[PYDTNN_POOL_SHAPE] = (pool_shape, pool_shape)
        #else: It mus be a Tuple[int, int], so it's okay

    return layers.AveragePool2D(**layer_args)
# --- END AvgPool2d --- #

def AdaptiveAvgPool2d(args: Dict[str, Any]) -> layers.AveragePool2D:
    # https://pytorch.org/docs/stable/generated/torch.nn.AdaptiveAvgPool2d.html#torch.nn.AdaptiveAvgPool2d
    # from torch.nn import AdaptiveAvgPool2d    
    # NOTE: "The output is of size H x W, for any input size. The number of output features is equal to the number of input planes." Source: PyTorch.      
    
    arguments = args[cons.ARGUMENTS]    
    _output_size = arguments[cons.PYTORCH_OUTPUT_SIZE] if cons.PYTORCH_OUTPUT_SIZE in arguments else None
    
    # TODO: Check if there is a way to avoid this (~terrorism~) weird patch.
    layer = layers.AveragePool2D()

    base_initialize = layer.initialize
    
    # NOTE: IMPORTANT if the base function's arguments change, it is necessary to change them here too.
    def new_initialize(prev_shape, need_dx=True):
        
        # Information used in order to make this function's operations:
        # https://stackoverflow.com/questions/58692476/what-is-adaptive-average-pooling-and-how-does-it-work 
        # https://stackoverflow.com/questions/53841509/how-does-adaptive-pooling-in-pytorch-work/63603993#63603993

        # Situation Example: 
        #   input_size: | type: <class 'tuple'> | value: (31, 31, 4)
        #   output_size: | type: <class 'list'> | value: [4, 4]

        input_size = prev_shape
        # It's possible that it cames with the format 4 ==> transform to [4,4] and operate.
        if _output_size is not None:
            output_size = (_output_size, _output_size) if isinstance(_output_size, int) else _output_size
        else:
            # Seems that, internally the format is: "PYDTNN_TENSOR_FORMAT_NHWC" [TODO: Check this affirmation] ==> 
            #   ==> The N dimension is implicit and the C dimension must ot be ignored
            output_size = input_size[:-1]           

        #layer.stride = input_size // output_size
        len_output_size = len(output_size)
        layer.stride = tuple([ input_size[i] // output_size[i] for i in range(len_output_size)])

        #layer.pool_shape = input_size - (output_size - 1) * layer.stride  
        layer.pool_shape = [0] * len_output_size
        for i in range(len_output_size):
            layer.pool_shape[i] = input_size[i] - (output_size[i] - 1) * layer.stride[i]
        layer.pool_shape = tuple(layer.pool_shape)

        layer.padding = 0

        # Calling directly "layer.initialize" will produce a recursive call (Remember: this function will be "layer.initialize")
        base_initialize(prev_shape = prev_shape, need_dx = need_dx)
    # -- END new_initialize -- #
    
    layer.initialize = new_initialize

    return layer
# --- END AdaptiveAvgPool2d --- #