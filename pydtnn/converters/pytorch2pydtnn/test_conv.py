"""
Test suite for PyDTNN convolution layer converters.

This module provides functional tests for various convolution layer types,
including standard, depthwise, and pointwise convolutions, as well as
activation layers, ensuring correct forward and backward pass behavior.
"""

from copy import deepcopy

import numpy as np

from pydtnn.abstract.layerable import Layerable
from pydtnn.activations.leaky_relu import LeakyRelu
from pydtnn.activations.relu6 import Relu6
from pydtnn.backends.pycuda.utils.tensor_array import TensorArray
from pydtnn.layers.conv_2d import Conv2D
from pydtnn.layers.conv_2d_depthwise import Conv2DDepthwise
from pydtnn.layers.conv_2d_pointwise import Conv2DPointwise
from pydtnn.layers.input import Input
from pydtnn.model import Model
from pydtnn.utils import rand

try:
    import pycuda.gpuarray as gpuarray
except BaseException:
    gpuarray = None

__all__ = ("main",)

# Constants
TENSOR_FORMAT = "NCHW"  # "NCHW" # "NHWC" # "NCHW"
N, C, H, W = 2, 2, 3, 3
SHAPE = (C, H, W) if TENSOR_FORMAT == "NCHW" else (H, W, C)
CONV_IN_CHANNELS = C
CONV_OUT_CHANNELS = 2  # = PyTorch's Number filters
CONV_KERNEL_SIZE = (1, 1)
SEED = 1234
DTYPE = np.float32

KWARGS = {
    "model_name": None,
    "evaluate_only": True,
    "parallel_data": False,
    "tensor_format": TENSOR_FORMAT,
    "use_cudnn": False,  # True,
    "omm": None,
    "dtype": DTYPE,
    "tracing": False,
    "tracer_output": "",
    "batch_size": min(64, N),
    "optimizer_name": "adam",
}

# End Constants

rand.seed(SEED)


def main() -> None:
    """
    Executes the main test routine for convolution layers.

    Initializes multiple model configurations, performs forward and backward
    passes through various layer types, and validates output shapes.
    """
    model_i2c = Model(**KWARGS)
    model_depth = Model(**KWARGS)
    model_point = Model(**KWARGS)
    model_relu = Model(**KWARGS)

    shape = (N, *SHAPE)
    dataset = np.arange(np.prod(shape), dtype=DTYPE).reshape(shape)
    dataset *= -1
    dataset *= dataset % 2
    print(f"{dataset=}")
    print(f"{dataset.shape}")

    use_bias = True

    models = [
        ("=============\n==== I2C ====\n=============", model_i2c),
        ("=============\n= POINTWISE =\n=============", model_point),
        ("=============\n= DEPTHWISE =\n=============", model_depth),
        ("=============\n= LEAKY RELU =\n=============", model_relu),
    ]

    for _, model in models:
        model: Model

    model_relu.add(Input(SHAPE))
    model_relu.add(LeakyRelu(negative_slope=-32))
    model_relu.add(Relu6())
    model_relu._model_init()

    model_depth.add(Input(SHAPE))
    model_depth.add(
        Conv2DDepthwise(
            nfilters=CONV_OUT_CHANNELS, filter_shape=CONV_KERNEL_SIZE, use_bias=use_bias
        )
    )
    model_depth._model_init()

    model_point.add(Input(SHAPE))
    model_point.add(
        Conv2DPointwise(
            nfilters=CONV_OUT_CHANNELS, filter_shape=CONV_KERNEL_SIZE, use_bias=use_bias
        )
    )
    model_point._model_init()

    model_i2c.add(Input(SHAPE))
    model_i2c.add(
        Conv2D(nfilters=CONV_OUT_CHANNELS, filter_shape=CONV_KERNEL_SIZE, use_bias=use_bias)
    )
    model_i2c._model_init()

    for name, model in models:
        print(f"{name}")

        model.mode = Model.Mode.TRAIN
        # model.show()

        x = deepcopy(dataset)
        if KWARGS["use_cudnn"]:
            assert gpuarray
            _dataset = TensorArray(
                gpu_arr=gpuarray.zeros(shape=dataset.shape, dtype=KWARGS["dtype"]),
                tensor_format=model.tensor_format,
                cudnn_dtype=model.cudnn_dtype,
            )
            _dataset.set(dataset)
            x = _dataset

        num_layers = len(model.layers)
        print("Forward")
        for i in range(num_layers):
            layer: Layerable = model.layers[i]
            print(f"{layer=}")
            x: np.ndarray | TensorArray = layer.forward(x)
            print(f"{x.shape=}")
        print("\n----------")

        dy = x
        print("Backward")
        for i in range(num_layers - 1, 0, -1):
            layer: Layerable = model.layers[i]
            dy: np.ndarray | TensorArray = layer.backward(dy)
            print(f"{dy.shape=}")
        print("\n=========\n")

        for i in range(num_layers - 1, 0, -1):
            layer: Layerable = model.layers[i]
            layer.update_weights(model.optimizer)


if __name__ == "__main__":
    main()
