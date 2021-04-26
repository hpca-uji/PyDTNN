"""
Unitary tests for GPU with different models' layers

For running all the tests quietly, execute the next command:
    python -um unittest pydtnn.tests.GPUModelsTestCase

For running all the tests verbosely, execute the next command:
    python -um unittest -v pydtnn.tests.GPUModelsTestCase

For running an individual test verbosely, execute the next command:
    python -um unittest -v pydtnn.tests.GPUModelsTestCase.test_name
"""

import sys
import unittest
import warnings

# noinspection PyUnresolvedReferences
import pycuda.gpuarray as gpuarray

from pydtnn.backends.gpu.tensor_gpu import TensorGPU
from pydtnn.model import Model
from pydtnn.tests import ConvGemmModelsTestCase
from pydtnn.tests.common import verbose_test


class Params:
    pass


class GPUModelsTestCase(ConvGemmModelsTestCase):
    """
    Tests that two models with different parameters lead to the same results
    """

    # Compares results between an XX model {self.model1_desc} and other {self.model1_desc}
    model1_desc = "using the CPU backend"
    model2_desc = "using the GPU backend"

    rtol_dict = {
        "AdditionBlock": 2e-4,
        "ConcatenationBlock": 2e-4,
    }

    atol_dict = {
        "AdditionBlock": 1e-2,
        "BatchNormalization": 2e-5,
        "ConcatenationBlock": 5e-3,
        "Conv2D": 4e-3,
    }

    @staticmethod
    def get_model2(model_name):
        # GPU model
        params = Params()
        params.model_name = model_name
        params.enable_gpu = True
        params.enable_cudnn_auto_conv_alg = True
        return Model(**vars(params))

    @staticmethod
    def copy_weights_and_biases(model1, model2):
        """
        Copy weights and biases from Model 1 to Model 2
        """
        for cpu_layer, gpu_layer in zip(model1.get_all_layers()[1:], model2.get_all_layers()[1:]):
            if len(cpu_layer.weights.shape) == 1:
                continue
            gpu_layer.weights_cpu = cpu_layer.weights.copy()
            if len(gpu_layer.weights_cpu):
                weights_gpu = gpuarray.to_gpu(gpu_layer.weights_cpu)
                gpu_layer.weights = TensorGPU(weights_gpu, gpu_layer.model.tensor_fmt,
                                              gpu_layer.model.cudnn_dtype, "filter")
            if gpu_layer.use_bias:
                if len(cpu_layer.biases.shape) == 1:
                    continue
                gpu_layer.biases_cpu = cpu_layer.biases.copy()
                if len(gpu_layer.biases_cpu):
                    biases_gpu = gpuarray.to_gpu(gpu_layer.biases_cpu)
                    gpu_layer.biases = TensorGPU(biases_gpu, gpu_layer.model.tensor_fmt,
                                                 gpu_layer.model.cudnn_dtype)

    @staticmethod
    def do_model2_forward_pass(model2, x1):
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
    def do_model2_backward_pass(model2, dx1):
        """
        Model 2 backward pass
        """
        dx2 = [None] * len(model2.layers)
        dx2[-1] = dx1[-1]
        for i, layer in reversed(list(enumerate(model2.layers[2:-1], 2))):
            if verbose_test():
                print(layer)
            try:
                model2.layers[i + 1].dx.ary.set(dx1[i + 1])
            except ValueError:
                warnings.warn(f"dx of model 1 {model2.layers[i + 1].canonical_name_with_id}"
                              f" is not ordered [dx.strides: {dx1[i + 1].strides}")
                model2.layers[i + 1].dx.ary.set(dx1[i + 1].copy())
            out = layer.backward(model2.layers[i + 1].dx)
            dx2[i] = out.ary.get()
        return dx2


if __name__ == '__main__':
    try:
        Model()
    except NameError:
        sys.exit(-1)
    unittest.main()
