import unittest
import warnings

import pycuda.gpuarray as gpuarray

from pydtnn.backends.gpu.utils.tensor_gpu import TensorGPU
from pydtnn.layers.conv_2d import Conv2D
from pydtnn.layers.layer import LayerError
from pydtnn.model import Model
from pydtnn.tests.model_common import ModelCommonTestCase
from pydtnn.tests.common import verbose_test, Params
from pydtnn.utils.tensor import TensorFormat


class ModelGpuTestCase(ModelCommonTestCase):
    """
    Tests that two models with different parameters lead to the same results
    """
    # NOTE: Delete parent test to prevent re-export and re-testing
    global ModelCommonTestCase
    del ModelCommonTestCase

    # Compares results between an XX model {self.model1_desc} and other {self.model1_desc}
    model1_desc = "using the CPU backend"
    model2_desc = "using the GPU backend"

    @staticmethod
    def get_model2(model_name: str) -> Model:
        # GPU model
        params = Params()
        params.model_name = model_name
        params.enable_gpu = True
        params.enable_cudnn_auto_conv_alg = True
        params.tensor_format = TensorFormat.NHWC.upper()
        params_dict = vars(params)
        try:
            model2 = Model(**params_dict)
        except LayerError as exc:
            raise unittest.SkipTest(f"Model {model_name} incompatible with {params_dict['dataset_name']}") from exc
        return model2

    @staticmethod
    def copy_weights_and_biases(model1: Model, model2: Model):
        """
        Copy weights and biases from Model 1 to Model 2
        """
        for cpu_layer, gpu_layer in zip(model1.get_all_layers()[1:], model2.get_all_layers()[1:]):
            if cpu_layer.weights is None:
                continue
            if isinstance(gpu_layer, Conv2D):
                if model2.tensor_format is TensorFormat.NHWC:
                    # TODO: check this.
                    gpu_layer.weights_cpu = cpu_layer.weights.transpose(3, 1, 2, 0).copy()
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

    @staticmethod
    def do_model2_forward_pass(model2: Model, x1: list[TensorGPU]) -> list[TensorGPU]:
        """
        Model 2 forward pass
        """
        x2 = [x1[0]]
        for i, layer in enumerate(model2.layers[1:], 1):
            if verbose_test():
                print(layer)
            try:
                model2.layers[i - 1].y.ary.set(x1[i - 1])
            except ValueError:
                warnings.warn(f"Output of model 1 {model2.layers[i - 1].canonical_name_with_id}"
                              f" is not ordered [x.strides: {x1[i - 1].strides}")
                model2.layers[i - 1].y.ary.set(x1[i - 1].copy())
            out = layer.forward(model2.layers[i - 1].y)
            x2.append(out.ary.get())
        return x2

    @staticmethod
    def do_model2_backward_pass(model2: Model, dx1: list[TensorGPU]) -> list[TensorGPU]:
        """
        Model 2 backward pass
        """
        dx2 = [None] * len(model2.layers)
        dx2[-1] = dx1[-1]
        for i, layer in reversed(list(enumerate(model2.layers[2:-1], 2))):
            if verbose_test():
                print(layer)
            try:
                model2.layers[i + 1].dx.ary.set(dx1[i + 1].reshape(model2.layers[i + 1].dx.ary.shape))
            except ValueError:
                warnings.warn(f"dx of model 1 {model2.layers[i + 1].canonical_name_with_id}"
                              f" is not ordered [dx.strides: {dx1[i + 1].strides}")
                model2.layers[i + 1].dx.ary.set(dx1[i + 1].reshape(model2.layers[i + 1].dx.ary.shape).copy())
            out = layer.backward(model2.layers[i + 1].dx)
            dx2[i] = out.ary.get()
        return dx2
