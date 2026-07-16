"""Module for testing PyDTNN model consistency between CPU and GPU backends."""

from __future__ import annotations

import logging
import unittest

import numpy as np
import pycuda.gpuarray as gpuarray  # type: ignore

from pydtnn import pycuda, supported_gpu
from pydtnn.abstract.layerable import Layerable
from pydtnn.backends.pycuda.utils.tensor_array import TensorArray
from pydtnn.layers.abstract.layer import LayerError
from pydtnn.layers.addition_block import AdditionBlock
from pydtnn.layers.concatenation_block import ConcatenationBlock
from pydtnn.layers.conv_2d import Conv2D
from pydtnn.model import Model
from pydtnn.tests.abstract.common import Params, verbose_test
from pydtnn.tests.abstract.model_common import ModelCommonTestCase
from pydtnn.utils.tensor import TensorFormat, format_transpose

__all__ = ("ModelGpuTestCase",)

logger = logging.getLogger(__name__)


@unittest.skipUnless(pycuda and supported_gpu, "requires GPU")
class ModelGpuTestCase(ModelCommonTestCase):
    """Test case for verifying model parity between CPU and GPU implementations."""

    global ModelCommonTestCase

    rtol_dict = ModelCommonTestCase.rtol_dict | {
        ConcatenationBlock: 1e-1,
        AdditionBlock: 1e-1,
        Conv2D: 1e-4,
    }
    atol_dict = ModelCommonTestCase.atol_dict | {
        ConcatenationBlock: 1e-1,
        AdditionBlock: 1e-1,
        Conv2D: 1e-4,
    }

    # NOTE: Delete parent test to prevent re-export and re-testing
    del ModelCommonTestCase

    # Compares results between an XX model {self.model1_desc} and other {self.model1_desc}
    model1_desc = "using the CPU backend"
    model2_desc = "using the GPU backend"

    def get_model2(self, model_name: str) -> Model:
        """
        Initialize and return a model configured for the GPU backend.

        Args:
            model_name: The name of the model to instantiate.

        Returns:
            A Model instance configured with the GPU backend.
        """
        # GPU model
        params = Params()
        params.model_name = model_name  # type: ignore
        params.backend = "gpu"
        params.use_cudnn = True  # type: ignore
        params.use_cudnn_auto_conv_algo = True  # type: ignore
        params.tensor_format = TensorFormat.NCHW.upper()
        params_dict = vars(params)
        try:
            model2 = Model(**params_dict)
        except LayerError as exc:
            raise unittest.SkipTest(
                f"Model {model_name} incompatible with {params_dict['dataset_name']}"
            ) from exc
        model2._model_init()
        return model2

    def copy_weights_and_biases(self, model1: Model, model2: Model) -> None:
        """
        Copy weights and biases from a CPU model to a GPU model.

        Args:
            model1: The source CPU model.
            model2: The destination GPU model.
        """
        for cpu_layer, gpu_layer in zip(model1.get_all_layers(), model2.get_all_layers()):
            if cpu_layer.weights is None:
                continue
            if isinstance(gpu_layer, Conv2D):
                if model2.tensor_format is TensorFormat.NHWC:
                    gpu_layer.weights_cpu = format_transpose(
                        cpu_layer.weights, "IHWO", "OHWI"
                    ).copy()
                else:
                    gpu_layer.weights_cpu = cpu_layer.weights.copy()
            else:
                gpu_layer.weights_cpu = cpu_layer.weights.copy()
            if gpu_layer.weights_cpu is not None:
                weights_gpu = gpuarray.to_gpu(gpu_layer.weights_cpu)
                gpu_layer.weights = TensorArray(
                    weights_gpu,
                    gpu_layer.model.tensor_format,
                    gpu_layer.model.cudnn_dtype,
                    TensorArray.TensorType.FILTER,
                )
            if gpu_layer.use_bias:
                if cpu_layer.biases is None:
                    continue

                gpu_layer.biases_cpu = cpu_layer.biases.copy()
                if gpu_layer.biases_cpu is not None:
                    biases_gpu = gpuarray.to_gpu(gpu_layer.biases_cpu)
                    gpu_layer.biases = TensorArray(
                        biases_gpu, gpu_layer.model.tensor_format, gpu_layer.model.cudnn_dtype
                    )

    def set_data_to_ary(
        self,
        ary: gpuarray,  # type: ignore
        data: np.ndarray,
        layer: Layerable,
    ) -> None:
        """
        Upload numpy data to a GPU array.

        Args:
            ary: The target GPU array.
            data: The source numpy array.
            layer: The layer associated with the data.
        """
        try:
            ary.set(data.copy())
        except ValueError as e:
            raise ValueError(
                f"Output of model 1 {layer.name_with_id} is not ordered [x.strides: {data.strides}"
            ) from e

    def do_model2_forward_pass(self, model2: Model, x1: list[np.ndarray]) -> list[np.ndarray]:
        """
        Execute a forward pass on the GPU model.

        Args:
            model2: The GPU model.
            x1: A list of input numpy arrays.

        Returns:
            A list of output numpy arrays from the forward pass.
        """
        x2 = [x1[0]]
        # Input layer
        layer = model2.layers[0]
        self.set_data_to_ary(layer.y, x1[0], layer)
        out = layer.forward(layer.y)
        x2.append(out.get())

        # The rest of the layers
        for i, layer in enumerate(model2.layers[1:], 1):
            if verbose_test():
                if verbose_test():
                    logger.info(layer)
            self.set_data_to_ary(model2.layers[i - 1].y, x1[i], layer)
            out = layer.forward(model2.layers[i - 1].y)
            x2.append(out.get())
        return x2

    def do_model2_backward_pass(self, model2: Model, dx1: list[np.ndarray]) -> list[np.ndarray]:
        """
        Execute a backward pass on the GPU model.

        Args:
            model2: The GPU model.
            dx1: A list of gradient numpy arrays.

        Returns:
            A list of gradient numpy arrays from the backward pass.
        """
        dx2 = [dx1[-1].copy()]

        layer = model2.layers[-1]
        self.set_data_to_ary(model2.layers[-1].dx, dx1[-1], layer)
        out = layer.backward(model2.layers[-1].dx)
        dx2.insert(0, out.get().copy())

        for i, layer in reversed(list(enumerate(model2.layers))[:-1]):
            if verbose_test():
                logger.info(layer)
            self.set_data_to_ary(model2.layers[i + 1].dx, dx1[i + 1], layer)
            out = layer.backward(model2.layers[i + 1].dx)
            dx2.insert(0, out.get().copy())
        return dx2
