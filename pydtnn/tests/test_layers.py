# import memray

from pydtnn.activations import *
from pydtnn.layers import *
from pydtnn.optimizers import *

from pydtnn.model import Model
from pydtnn.utils import random

import numpy as np
from time import time

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    # noinspection PyUnresolvedReferences
    import pycuda.gpuarray as gpuarray
from pydtnn.backends.gpu import TensorGPU

# setting random seed
SEED = 1234
random.seed(SEED)
# ---

N = 100
C = 3
H = 524
W = 524
FORMAT = "NHWC"
SHAPE = (C, H, W) if FORMAT == "NCHW" else (H, W, C)
NUM_REPETITIONS = 10

KWARGS = {
    "model_name": None,
    # "dataset": None,
    # "dataset_name": None,
    "evaluate_only": True,
    "parallel": "data",
    "tensor_format": FORMAT,  # "NCHW" # "NHWC",
    "loss_func": "categorical_cross_entropy",
    "enable_gpu": False,  # False, #True,
    "omm": None,
    "dtype": np.float32,
    "tracing": False,
    "tracer_output": "",
    "batch_size": N
}

ignore_model = Model(**KWARGS)

list_layers = [
    # ("AdaptiveAveragePool2D",AdaptiveAveragePool2D(output_shape=(3, 3))), # Not in older versions
    # ("AveragePool2D",AveragePool2D()),
    # ("BatchNormalization",BatchNormalization()),
    ("Conv2D_STANDARD", Conv2D(grouping=Conv2D.Grouping.STANDARD)),
    ("Conv2D_DEPTHWISE", Conv2D(grouping=Conv2D.Grouping.DEPTHWISE)),
    ("Conv2D_POINTWISE", Conv2D(grouping=Conv2D.Grouping.POINTWISE)),
    # ("Dropout",Dropout()),
    # ("FC",FC()),
    # ("Flatten",Flatten()),
    # ("MaxPool2D",MaxPool2D()),
    # -----
    # ("BatchNormalizationRelu",BatchNormalizationRelu()),

]
list_activations = [
    # ("Sigmoid", Sigmoid()),
    # ("Relu", Relu()),
    # ("Relu6", Relu6()), # Not in older versions.
    # ("LeakyRelu", LeakyRelu()), # Not in older versions.
    # ("Tanh", Tanh()),
    # ("Arctanh", Arctanh()),
    # ("Log", Log()),
    # ("Softmax", Softmax())
]

# list_optimizers = [Adam(), Nadam(), RMSProp(), SGD()]
addition_test_layers = ("AdditionBlock", AdditionBlock([Conv2D(grouping=Conv2D.Grouping.STANDARD), BatchNormalization()], [Conv2D(grouping=Conv2D.Grouping.STANDARD)]))
concatenation_test_layers = ("ConcatenationBlock", ConcatenationBlock([Conv2D(grouping=Conv2D.Grouping.STANDARD), BatchNormalization()], [Conv2D(grouping=Conv2D.Grouping.STANDARD)]))

dict_test: dict[str, Activation | tuple[str, Layer]] = {
    "Layers": list_layers,
    "Activations": list_activations,
}


def test_keras(_x: np.ndarray):

    model = Model(**KWARGS)
    model.add(Input(SHAPE))
    model.add(Conv2D(grouping=Conv2D.Grouping.STANDARD, nfilters=3))
    model.mode = Model.Mode.TRAIN
    model._initialize()

    x = np.copy(_x)

    for layer in model.layers:
        x_pydtnn = layer.forward(x)

    print(f"x_pydtnn.max:\t{x_pydtnn.max()}")
    print(f"x_pydtnn.min:\t{x_pydtnn.min()}")

    print(f"x_pydtnn:\n{x_pydtnn}")
# ---

def test_layers_activations(_x: np.ndarray, opt: Optimizer) -> None:
    # Testing Layers and activations:
    for test in dict_test.keys():
        print(f"=====\nTesting {test}\n=====")
        for name, test_elem in dict_test[test]:
            print(f"- Testing: {name}")
            model = Model(**KWARGS)
            model.add(Input(SHAPE))
            if name == "FC":
                model.add(Flatten())
            model.add(test_elem)
            model.mode = Model.Mode.TRAIN
            model._initialize()
            opt.initialize(model.get_all_layers(model.layers))

            x = np.copy(_x)

            if KWARGS["enable_gpu"]:
                x = TensorGPU(gpuarray.to_gpu(x), model.tensor_format, model.cudnn_dtype)

            t_forward = 0.0
            t_backward = 0.0
            t_opt = 0.0

            for i in range(NUM_REPETITIONS):
                if True:
                    # with memray.Tracker(f"./z_memray/{KWARGS['tensor_format']}/fwd/{name}_{i}.bin", native_traces=True):

                    t = time()
                    for layer in model.layers:
                        x = layer.forward(x)
                    t_forward += time() - t

                if not KWARGS["enable_gpu"]:
                    x = x.copy()

                if True:
                    # with memray.Tracker(f"./z_memray/{KWARGS['tensor_format']}/bwd/{name}_{i}.bin", native_traces=True):
                    t = time()
                    for layer in reversed(model.layers):
                        x = layer.backward(x)
                    t_backward += time() - t

                if True:
                    # with memray.Tracker(f"./z_memray/{KWARGS['tensor_format']}/opt/{name}_{opt.canonical_name}_{i}.bin", native_traces=True):
                    t = time()
                    for layer in reversed(model.layers):
                        layer.update_weights(opt)
                    t_opt += time() - t

            print(f"Forward time mean: {t_forward / NUM_REPETITIONS:2f} s")
            print(f"Backward time mean: {t_backward / NUM_REPETITIONS:2f} s")
            print(f"Optimizer time mean: {t_opt / NUM_REPETITIONS:2f} s")

            print(f"------------")
# --- END test_layers_activations --- #


def test_add_concat(_x: np.ndarray, opt: Optimizer) -> None:
    # Testing Addition and Concatenation layers
    for test, layer in [addition_test_layers, concatenation_test_layers]:
        print(f"=====\nTesting the: {test}\n=====")
        model = Model(**KWARGS)
        model.add(Input(SHAPE))
        model.add(layer)
        model.mode = Model.Mode.TRAIN
        model._initialize()
        opt.initialize(model.get_all_layers(model.layers))

        x = np.copy(_x)
        t_forward = 0
        t_backward = 0
        t_opt = 0

        if KWARGS["enable_gpu"]:
            x = TensorGPU(gpuarray.to_gpu(x), model.tensor_format, model.cudnn_dtype)

        for i in range(NUM_REPETITIONS):
            if True:
                # with memray.Tracker(f"./z_memray/{KWARGS['tensor_format']}/fwd/{test}_{i}.bin", native_traces=True):

                t = time()
                for layer in model.layers:
                    x = layer.forward(x)
                t_forward += time() - t

            if not KWARGS["enable_gpu"]:
                x = x.copy()

            if True:
                # with memray.Tracker(f"./z_memray/{KWARGS['tensor_format']}/bwd/{test}_{i}.bin", native_traces=True):
                t = time()
                for layer in reversed(model.layers):
                    x = layer.backward(x)
                t_backward += time() - t
            if True:
                # with memray.Tracker(f"./z_memray/{KWARGS['tensor_format']}/opt/{test}_{opt.canonical_name}_{i}.bin", native_traces=True, ):
                t = time()
                for layer in reversed(model.layers):
                    layer.update_weights(opt)
                t_opt += time() - t

        print(f"Forward time: {t_forward / NUM_REPETITIONS:2f} s")
        print(f"Backward time: {t_backward / NUM_REPETITIONS:2f} s")
        print(f"Optimizer time: {t_opt / NUM_REPETITIONS:2f} s")

        print(f"------------")
# --- END test_add_concat --- #


def main():
    shape = (N, *SHAPE)
    quarter_elements = np.prod(shape) / 4
    _x_p: np.ndarray = np.arange(np.ceil(quarter_elements), dtype=KWARGS["dtype"])
    _x_n: np.ndarray = np.arange(np.ceil(quarter_elements), dtype=KWARGS["dtype"]) * 1
    _x_p_ir: np.ndarray = np.arange(np.floor(quarter_elements), dtype=KWARGS["dtype"]) / 3
    _x_n_ir: np.ndarray = np.arange(np.floor(quarter_elements), dtype=KWARGS["dtype"]) * (1 / 3)
    _x = np.concatenate([_x_p, _x_n, _x_p_ir, _x_n_ir], dtype=KWARGS["dtype"]).reshape(shape)

    opt = SGD()

    print(f"dataset shape: {_x.shape}")
    # test_keras(_x)
    # test_torch(_x)
    test_layers_activations(_x, opt)
    # test_add_concat(_x, opt)


if __name__ == "__main__":
    main()
