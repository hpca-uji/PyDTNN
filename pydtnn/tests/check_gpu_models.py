"""
Unitary tests for GPU with different models' layers

For running all the tests quietly, execute the next command:
    python -um unittest pydtnn.tests.CheckGPUModels

For running all the tests verbosely, execute the next command:
    python -um unittest -v pydtnn.tests.CheckGPUModels

For running an individual test verbosely, execute the next command:
    python -um unittest -v pydtnn.tests.CheckGPUModels.test_name
"""

import sys
import unittest
import warnings

# noinspection PyUnresolvedReferences
import pycuda.gpuarray as gpuarray

from pydtnn.backends.gpu.tensor_gpu import TensorGPU
from pydtnn.model import Model
from pydtnn.tests import CheckConvGemmModels
from pydtnn.tests.common import verbose_test
from pydtnn.utils import PYDTNN_TENSOR_FORMAT_NCHW, PYDTNN_TENSOR_FORMAT_NHWC
from pydtnn import losses


class Params:
    pass


class CheckGPUModels(CheckConvGemmModels):
    """
    Tests that two models with different parameters lead to the same results
    """

    # Compares results between an XX model {self.model1_desc} and other {self.model1_desc}
    model1_desc = "using the CPU backend"
    model2_desc = "using the GPU backend"

    rtol_dict = {
        "AdditionBlock": 1e-2,
        "ConcatenationBlock": 1e-2,
    }

    atol_dict = {
        "AdditionBlock": 1e-2,
        "BatchNormalization": 6e-5,
        "ConcatenationBlock": 1e-2,
        "Conv2D": 4e-3,
        "FC": 1e-5,
    }

    @staticmethod
    def get_model1_and_loss_func(model_name):
        # CPU model with no convGemm
        params = Params()
        params.model_name = model_name
        params.enable_conv_gemm = False
        params.conv_gemm_cache = False
        params.tensor_format = "NHWC"
        model1 = Model(**vars(params))
        # loss function
        loss = model1.loss_func
        local_batch_size = model1.batch_size
        loss_func = getattr(losses, loss)(shape=(local_batch_size, *model1.layers[-1].shape), model=model1)
        return model1, loss_func

    @staticmethod
    def get_model2(model_name):
        # GPU model
        params = Params()
        params.model_name = model_name
        params.enable_gpu = True
        params.enable_cudnn_auto_conv_alg = True
        params.tensor_format = "NHWC"
        return Model(**vars(params))

    @staticmethod
    def copy_weights_and_biases(model1, model2):
        """
        Copy weights and biases from Model 1 to Model 2
        """
        for cpu_layer, gpu_layer in zip(model1.get_all_layers()[1:], model2.get_all_layers()[1:]):
            if len(cpu_layer.weights.shape) == 1:
                continue
            if "Conv2D" in type(gpu_layer).__name__:
                if model2.tensor_format == PYDTNN_TENSOR_FORMAT_NHWC:
                    gpu_layer.weights_cpu = cpu_layer.weights.transpose(3, 1, 2, 0).copy()
                else:
                    gpu_layer.weights_cpu = cpu_layer.weights.copy()
            else:
                gpu_layer.weights_cpu = cpu_layer.weights.copy()
            if len(gpu_layer.weights_cpu):
                weights_gpu = gpuarray.to_gpu(gpu_layer.weights_cpu)
                gpu_layer.weights = TensorGPU(weights_gpu, gpu_layer.model.tensor_format,
                                              gpu_layer.model.cudnn_dtype, "filter")
            if gpu_layer.use_bias:
                if len(cpu_layer.biases.shape) == 1:
                    continue
                gpu_layer.biases_cpu = cpu_layer.biases.copy()
                if len(gpu_layer.biases_cpu):
                    biases_gpu = gpuarray.to_gpu(gpu_layer.biases_cpu)
                    gpu_layer.biases = TensorGPU(biases_gpu, gpu_layer.model.tensor_format,
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
                model2.layers[i + 1].dx.ary.set(dx1[i + 1].reshape(model2.layers[i + 1].dx.ary.shape))
            except ValueError:
                warnings.warn(f"dx of model 1 {model2.layers[i + 1].canonical_name_with_id}"
                              f" is not ordered [dx.strides: {dx1[i + 1].strides}")
                model2.layers[i + 1].dx.ary.set(dx1[i + 1].reshape(model2.layers[i + 1].dx.ary.shape).copy())
            out = layer.backward(model2.layers[i + 1].dx)
            dx2[i] = out.ary.get()
        return dx2


if __name__ == '__main__':
    try:
        Model()
    except NameError:
        sys.exit(-1)
    unittest.main()
