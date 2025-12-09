import numpy as np
from pydtnn.model import Model
from pydtnn.layers.input import Input
from pydtnn.layers.layer import Layer
from pydtnn.activations.leaky_relu import LeakyRelu
from pydtnn.activations.relu6 import Relu6
from copy import deepcopy

from pydtnn.layers.conv_2d import Conv2D
from pydtnn.utils import random

from pydtnn.backends.gpu.utils.tensor_gpu import TensorGPU
try:
    import pycuda.gpuarray as gpuarray
    from pydtnn.libs import libcudnn as cudnn
except BaseException:
    pass

# Constants #
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
    "parallel": "data",
    "tensor_format": TENSOR_FORMAT,
    "enable_gpu": False,  # True,
    "omm": None,
    "dtype": DTYPE,
    "tracing": False,
    "tracer_output": "",
    "batch_size": min(64, N),
    "optimizer_name": "adam",
}

# End Constants #

random.seed(SEED)


def main():
    model_I2C = Model(**KWARGS)
    model_DEPTH = Model(**KWARGS)
    model_POINT = Model(**KWARGS)
    model_RELU = Model(**KWARGS)

    shape = (N, *SHAPE)
    dataset = np.arange(np.prod(shape), dtype=DTYPE).reshape(shape)
    dataset *= -1
    dataset *= dataset % 2
    print(f"{dataset=}")
    print(f"{dataset.shape}")

    use_bias = True

    models = [
        ("=============\n==== I2C ====\n=============", model_I2C),
        ("=============\n= POINTWISE =\n=============", model_POINT),
        ("=============\n= DEPTHWISE =\n=============", model_DEPTH),
        ("=============\n= LEAKY RELU =\n=============", model_RELU)
    ]

    for _, model in models:
        model: Model
        model.dataset = dataset

    model_RELU.add(Input(SHAPE, is_shape_in_format=True))
    model_RELU.add(LeakyRelu(negative_slope=-32))
    model_RELU.add(Relu6())
    model_RELU._initialize()

    model_DEPTH.add(Input(SHAPE, is_shape_in_format=True))
    model_DEPTH.add(Conv2D(nfilters=CONV_OUT_CHANNELS, filter_shape=CONV_KERNEL_SIZE, grouping=Conv2D.Grouping.DEPTHWISE, use_bias=use_bias))
    model_DEPTH._initialize()

    model_POINT.add(Input(SHAPE, is_shape_in_format=True))
    model_POINT.add(Conv2D(nfilters=CONV_OUT_CHANNELS, filter_shape=CONV_KERNEL_SIZE, grouping=Conv2D.Grouping.POINTWISE, use_bias=use_bias))
    model_POINT._initialize()

    model_I2C.add(Input(SHAPE, is_shape_in_format=True))
    model_I2C.add(Conv2D(nfilters=CONV_OUT_CHANNELS, filter_shape=CONV_KERNEL_SIZE, use_bias=use_bias))
    model_I2C._initialize()

    for name, model in models:
        print(f"{name}")

        model.mode = Model.Mode.TRAIN
        # model.show()

        x = deepcopy(dataset)
        if KWARGS["enable_gpu"]:
            _dataset = TensorGPU(
                gpu_arr=gpuarray.empty(shape=dataset.shape, dtype=KWARGS["dtype"]),
                tensor_format=model.tensor_format, cudnn_dtype=model.cudnn_dtype)
            _dataset.ary.set(dataset)
            x = _dataset

        num_layers = len(model.layers)
        print("Forward")
        for i in range(num_layers):
            layer: Layer = model.layers[i]
            print(f"{layer=}")
            x: np.ndarray | TensorGPU = layer.forward(x)
            print(f"{x.shape=}")
        print("\n----------")

        dy = x
        print("Backward")
        for i in range(num_layers - 1, 0, -1):
            layer: Layer = model.layers[i]
            dy: np.ndarray | TensorGPU = layer.backward(dy)
            print(f"{dy.shape=}")
        print("\n=========\n")

        for i in range(num_layers - 1, 0, -1):
            layer: Layer = model.layers[i]
            layer.update_weights(model.optimizer)


if __name__ == "__main__":
    main()
