# _______________________________________________________________________________________________________________
# In this file must be implemented only the translation of PyTorch Pooling layers to its PyDTNN equivalent.
# _______________________________________________________________________________________________________________

# Typing related (or non important) imports
from typing import *
from inspect import stack # This is only in order to get the function's name

# Functionality imports
from pydtnn import layers
import pydtnn.translators.pytorch2pydtnn.constats as cons
import numpy as np
from pydtnn.utils import decode_tensor, encode_tensor

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

    cons.print_dict(args[cons.ARGUMENTS], "args[cons.ARGUMENTS]")
    cons.print_dict(layer_args, "layer_args")

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
        #else: It must be a Tuple[int, int], so it's okay

    cons.print_dict(args[cons.ARGUMENTS], "args[cons.ARGUMENTS]")
    cons.print_dict(layer_args, "layer_args")

    return layers.AveragePool2D(**layer_args)
# --- END AvgPool2d --- #

def AdaptiveAvgPool2d(args: Dict[str, Any]) -> layers.AveragePool2D:
    # https://pytorch.org/docs/stable/generated/torch.nn.AdaptiveAvgPool2d.html#torch.nn.AdaptiveAvgPool2d
    # from torch.nn import AdaptiveAvgPool2d    
    # NOTE: "The output is of size H x W, for any input size. The number of output features is equal to the number of input planes." Source: PyTorch.      
    print(f"Layer: {stack()[0].function}")

    arguments = args[cons.ARGUMENTS]    
    output_size = arguments[cons.PYTORCH_OUTPUT_SIZE] if cons.PYTORCH_OUTPUT_SIZE in arguments else None

    print("AAA")
    cons.print_dict(arguments, "arguments")
    print(f"output_size: {output_size}")    

    # TODO: Check if there is a way to avoid this (~terrorism~) weird patch.
    layer = layers.AveragePool2D()

    base_initialize = layer.initialize
    
    # NOTE: IMPORTANT if the base function's arguments change, it is necessary to change them here too.
    def new_initialize(prev_shape, need_dx=True, *, output_size = output_size, layer = layer, base_initialize = base_initialize):
        
        print("AAA")
        print(f"output_size: {output_size}")

        layer.hi, layer.wi, layer.ci = decode_tensor(prev_shape, layer.model.tensor_format)
        if output_size is not None:
            layer.ho, layer.wo = (output_size, output_size) if isinstance(output_size, int) else output_size
        else:
            layer.ho = layer.hi
            layer.wo = layer.wi
        layer.co = layer.ci

        layer.padding = 0
        layer.dilation = 1

        layer.vpadding, layer.hpadding = (layer.padding, layer.padding)
        layer.vdilation, layer.hdilation = (layer.dilation, layer.dilation)

        # Unknown values: pool_shape (kh, kw) and stride (vstride, hstride)

        # Getting (and setting) the pool_shape (kh, kw)
        layer.kh = layer.hi // layer.ho
        layer.kw = layer.wi // layer.wo

        print(f"layer.ci: {layer.ci}")
        print(f"layer.hi: {layer.hi}")
        print(f"layer.wi: {layer.wi}")
        print(f"layer.co: {layer.co}")
        print(f"layer.ho: {layer.ho}")
        print(f"layer.wo: {layer.wo}")
        print(f"layer.kh: {layer.kh}")
        print(f"layer.kw: {layer.kw}")
        print(f"layer.vpadding: {layer.vpadding}")
        print(f"layer.hpadding: {layer.hpadding}")
        print(f"layer.vdilation: {layer.vdilation}")
        print(f"layer.hdilation: {layer.hdilation}")

        # Getting (and setting) the stride (vstride, hstride)        
        # Base formula: self.ho = (self.hi + 2 * self.vpadding - self.vdilation * (self.kh - 1) - 1) // self.vstride + 1
        layer.vstride = (layer.hi + 2 * layer.vpadding - layer.vdilation * (layer.kh - 1) - 1) // (layer.ho - 1)
        # Base formula: self.wo = (self.wi + 2 * self.hpadding - self.hdilation * (self.kw - 1) - 1) // self.hstride + 1
        layer.hstride = (layer.wi + 2 * layer.hpadding - layer.hdilation * (layer.kw - 1) - 1) // (layer.wo - 1)
        

        print(f"layer.vstride: {layer.vstride}")
        print(f"layer.hstride: {layer.hstride}")

        #base_initialize(prev_shape, need_dx)
    # -- END new_initialize -- #
    
    layer.initialize = new_initialize

    return layer
# --- END AdaptiveAvgPool2d --- #


# NOTE: IMPORTANT if the base function's arguments change, it is necessary to change them here too.
    def new_initialize(prev_shape, need_dx=True, *, _output_size = _output_size, layer = layer, base_initialize = base_initialize):
        
        print("AAAAA")
        # Information used in order to make this function's operations:
        # https://stackoverflow.com/questions/58692476/what-is-adaptive-average-pooling-and-how-does-it-work 
        # https://stackoverflow.com/questions/53841509/how-does-adaptive-pooling-in-pytorch-work/63603993#63603993

        # Situation Example: 
        #   input_size: | type: <class 'tuple'> | value: (31, 31, 4)
        #   output_size: | type: <class 'list'> | value: [4, 4]

        _output_size = layer.shape
        print(f"_output_size: {_output_size}")

        # TODO: Mirar cómo llegan las cosas en el Flatten de CPU para ver si esto tiene sentido
        input_size = prev_shape
        # It's possible that it cames with the format 4 ==> transform to [4,4] and operate.
        if _output_size is not None:
            output_size = (_output_size, _output_size) if isinstance(_output_size, int) else _output_size
        else:
            output_size = input_size
        
        layer.padding= 0 
        layer.stride= 2 
        layer.dilation= 1
        layer.pool_shape = (2, 2)

        len_output_size = len(output_size)        
        #layer.stride = tuple([ input_size[i] // output_size[i] for i in range(len_output_size)])
        layer.pool_shape = tuple([ input_size[i] // output_size[i] for i in range(len_output_size)])
        layer.stride = tuple(((input_size[i] - layer.pool_shape[i]) // (output_size[i] - 1)) if output_size[i] > 1 
                             else layer.pool_shape[i]                               
                             for i in range(len_output_size))
        # layer.pool_shape = [0] * len_output_size        
        # for i in range(len_output_size):
        #     layer.pool_shape[i] = input_size[i] - (output_size[i] - 1) * layer.stride[i]
        # layer.pool_shape = tuple(layer.pool_shape)
        layer.pool_shape = tuple(input_size[i] - (output_size[i] - 1) * layer.stride[i] for i in range(len_output_size))        
        layer.padding = 0

        # TODO: 
        a = r"Ni idea de cómo hacer esto."
        "IDEA: Pasar de la variable, ver el tamaño de la capa actual, ver el tamaño de la capa anterior y calcularlo todo así"
        """
        x.shape: (64, 512, 2, 2)
        self.shape: (512, 1, 1)
        self.prev_shape: (512, 2, 2)
        self.pool_shape: (2, 2)
        self.padding: 0
        self.stride: 2
        self.dilation: 1
        self.co: 512
        self.ho: 1
        self.wo: 1
        
        """

        print(f"layer: {layer.shape}")
        print(f"input_size: {input_size}")
        print(f"_output_size: {_output_size}")
        print(f"output_size: {output_size}")
        print(f"layer.stride: {layer.stride}")
        print(f"layer.stride: {layer.stride}")
        print(f"layer.pool_shape: {layer.pool_shape}")
        print("---\n")

        # Calling directly "layer.initialize" will produce a recursive call (Remember: this function will be "layer.initialize")
        base_initialize(prev_shape = prev_shape, need_dx = need_dx)
    # -- END new_initialize -- #
    
    layer.initialize = new_initialize






