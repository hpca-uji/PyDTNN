import unittest
import numpy as np

import pycuda.gpuarray as gpuarray # type: ignore

from pydtnn.backends.gpu.utils.tensor_gpu import TensorGPU
from pydtnn.layer_base import LayerBase
from pydtnn.layers.addition_block import AdditionBlock
from pydtnn.layers.concatenation_block import ConcatenationBlock
from pydtnn.layers.conv_2d import Conv2D
from pydtnn.layers.layer import LayerError
from pydtnn.model import Model
from pydtnn.tests.abstract.model_common import ModelCommonTestCase
from pydtnn.tests.abstract.common import verbose_test, Params
from pydtnn.utils.tensor import TensorFormat, format_transpose


class ModelGpuTestCase(ModelCommonTestCase):
    """
    Tests that two models with different parameters lead to the same results
    """
    global ModelCommonTestCase

    rtol_dict = ModelCommonTestCase.rtol_dict | {ConcatenationBlock: 1e-1, AdditionBlock: 1e-1, Conv2D: 1e-4}
    atol_dict = ModelCommonTestCase.atol_dict | {ConcatenationBlock: 1e-1, AdditionBlock: 1e-1, Conv2D: 1e-4}

    # NOTE: Delete parent test to prevent re-export and re-testing
    del ModelCommonTestCase

    # Compares results between an XX model {self.model1_desc} and other {self.model1_desc}
    model1_desc = "using the CPU backend"
    model2_desc = "using the GPU backend"

    def get_model2(self, model_name: str) -> Model:
        # GPU model
        params = Params()
        params.model_name = model_name  # type: ignore
        params.enable_gpu = True  # type: ignore
        params.enable_cudnn_auto_conv_alg = True  # type: ignore
        params.tensor_format = TensorFormat.NHWC.upper()
        params_dict = vars(params)
        try:
            model2 = Model(**params_dict)
        except LayerError as exc:
            raise unittest.SkipTest(f"Model {model_name} incompatible with {params_dict['dataset_name']}") from exc
        return model2

    def copy_weights_and_biases(self, model1: Model, model2: Model):
        """
        Copy weights and biases from Model 1 to Model 2
        """
        for cpu_layer, gpu_layer in zip(model1.get_all_layers(), model2.get_all_layers()):
            if cpu_layer.weights is None:
                continue
            if isinstance(gpu_layer, Conv2D):
                if model2.tensor_format is TensorFormat.NHWC:
                    gpu_layer.weights_cpu = format_transpose(cpu_layer.weights, "IHWO", "OHWI").copy()
                else:
                    gpu_layer.weights_cpu = cpu_layer.weights.copy()
            else:
                gpu_layer.weights_cpu = cpu_layer.weights.copy()
            if gpu_layer.weights_cpu is not None:
                weights_gpu = gpuarray.to_gpu(gpu_layer.weights_cpu)
                gpu_layer.weights = TensorGPU(weights_gpu, gpu_layer.model.tensor_format,
                                              gpu_layer.model.cudnn_dtype, TensorGPU.TensorTypeEnum.FILTER)
            if gpu_layer.use_bias:
                if cpu_layer.biases is None:
                    continue

                gpu_layer.biases_cpu = cpu_layer.biases.copy()
                if gpu_layer.biases_cpu is not None:
                    biases_gpu = gpuarray.to_gpu(gpu_layer.biases_cpu)
                    gpu_layer.biases = TensorGPU(biases_gpu, gpu_layer.model.tensor_format,
                                                 gpu_layer.model.cudnn_dtype)

    def set_data_to_ary(self, ary: "gpuarray",  # type: ignore
                        data: np.ndarray, layer: LayerBase) -> None:
        try:
            ary.set(data.copy())
        except ValueError as e:
            raise ValueError(f"Output of model 1 {layer.name_with_id}" \
                             f" is not ordered [x.strides: {data.strides}") from e
    # ----

    def do_model2_forward_pass(self, model2: Model, x1: list[np.ndarray]) -> list[np.ndarray]:
        """
        Model 2 forward pass
        """
        x2 = [x1[0]]
        # Input layer
        layer = model2.layers[0]
        self.set_data_to_ary(layer.y.ary, x1[0], layer)
        out = layer.forward(layer.y)
        x2.append(out.ary.get())

        # The rest of the layers
        for i, layer in enumerate(model2.layers[1:], 1):
            if verbose_test():
                print(layer)
            self.set_data_to_ary(model2.layers[i - 1].y.ary, x1[i], layer)
            out = layer.forward(model2.layers[i - 1].y)
            x2.append(out.ary.get())
        return x2

    def do_model2_backward_pass(self, model2: Model, dx1: list[np.ndarray]) -> list[np.ndarray]:
        """
        Model 2 backward pass
        """
        dx2 = [dx1[-1].copy()]

        layer = model2.layers[-1]
        self.set_data_to_ary(model2.layers[-1].dx.ary, dx1[-1], layer)
        out = layer.backward(model2.layers[-1].dx)
        dx2.insert(0, out.ary.get().copy())

        for i, layer in reversed(list(enumerate(model2.layers))[:-1]):
            if verbose_test():
                print(layer)
            self.set_data_to_ary(model2.layers[i + 1].dx.ary, dx1[i + 1], layer)
            out = layer.backward(model2.layers[i + 1].dx)
            dx2.insert(0, out.ary.get().copy())
        return dx2
