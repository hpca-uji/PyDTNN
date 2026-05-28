"""
PyCUDA backend implementation for 2D Convolutional layers.
"""

import logging
from typing import Any

import numpy as np
import pycuda.driver as drv  # type: ignore
from pycuda import gpuarray  # type: ignore

from pydtnn.backends.pycuda.layers.abstract.layer import LayerPycuda
from pydtnn.backends.pycuda.utils.tensor_array import TensorArray
from pydtnn.layers.conv_2d import Conv2D
from pydtnn.utils.constants import ArrayShape, Parameters
from pydtnn.utils.performance_models import matmul_time
from pydtnn.utils.tensor import TensorFormat

__all__ = ("AbstractConv2DPycuda",)

logger = logging.getLogger(__name__)


class AbstractConv2DPycuda(Conv2D[TensorArray], LayerPycuda):
    """
    Abstract base class for 2D Convolutional layers using the PyCUDA backend.
    """

    def __init__(self, *args, **kwargs):
        """
        Initializes the abstract PyCUDA 2D convolution layer.
        """
        super().__init__(*args, **kwargs)

        # The following attributes will be initalized later.
        self.fwd_algo: int = None  # type: ignore
        self.bwd_dw_algo: int = None  # type: ignore
        self.bwd_dx_algo: int = None  # type: ignore
        self.conv_desc = None

    def _model_init(self, prev_shape: ArrayShape, x: TensorArray) -> None:
        """
        Initializes model parameters, GPU memory, and performance metrics.

        Args:
            prev_shape: The shape of the input tensor.
            x: The input tensor array.
        """
        super()._model_init(prev_shape, x)

        self.stream_2 = drv.Stream()

        self.weights_cpu = self.weights_initializer(self.weights_shape, self.model.dtype)
        weights_gpu = gpuarray.to_gpu(self.weights_cpu)
        self.weights = TensorArray(
            weights_gpu,
            self.model.tensor_format,
            self.model.cudnn_dtype,
            TensorArray.TensorType.FILTER,
        )
        self.memory_used += self.weights.nbytes

        # Biases
        if self.use_bias:
            biases_shape = self.model.encode_shape((1, self.co, 1, 1))
            self.biases_cpu = self.biases_initializer(biases_shape, self.model.dtype)
            biases_gpu = gpuarray.to_gpu(self.biases_cpu)
            self.biases = TensorArray(biases_gpu, self.model.tensor_format, self.model.cudnn_dtype)
            self.memory_used += self.biases.nbytes

        self.fwd_time = matmul_time(
            m=self.co,
            n=(self.model.batch_size * self.ho * self.wo),
            k=(self.ci * self.kh * self.kw),
            cpu_speed=self.model.cpu_speed,
            memory_bw=self.model.memory_bw,
            dtype=self.model.dtype,
        )  # type: ignore (It is correct.)
        self.bwd_time = matmul_time(
            m=self.co,
            n=(self.ci * self.kh * self.kw),
            k=(self.model.batch_size * self.ho * self.wo),
            cpu_speed=self.model.cpu_speed,
            memory_bw=self.model.memory_bw,
            dtype=self.model.dtype,
        ) + matmul_time(
            m=(self.ci * self.kh * self.kw),
            n=(self.model.batch_size * self.ho * self.wo),
            k=self.co,
            cpu_speed=self.model.cpu_speed,
            memory_bw=self.model.memory_bw,
            dtype=self.model.dtype,
        )  # type: ignore (It is correct.)

        if self.model.gpudirect:
            bias_tensor_type = TensorArray.TensorType.FILTER
            _drv = drv
        else:
            bias_tensor_type = TensorArray.TensorType.TENSOR
            _drv = None

        # Derivative dw and derivative db
        self.dw_cpu, self.dw = TensorArray.new(
            self.weights.shape,
            self.model.dtype,
            tensor_format=self.model.tensor_format,
            cudnn_dtype=self.model.cudnn_dtype,
            gpudirect=self.model.gpudirect,
            tensor_type=TensorArray.TensorType.FILTER,
            drv=_drv,
        )
        self.memory_used += self.dw.nbytes

        if self.use_bias:
            self.biases: TensorArray
            self.db_cpu, self.db = TensorArray.new(
                self.biases.shape,
                self.model.dtype,
                tensor_format=self.model.tensor_format,
                cudnn_dtype=self.model.cudnn_dtype,
                gpudirect=self.model.gpudirect,
                tensor_type=bias_tensor_type,
                drv=_drv,
            )
            self.memory_used += self.db.nbytes

    def forward(self, x: TensorArray) -> TensorArray:
        """
        Performs the forward pass. Must be implemented by subclasses.

        Args:
            x: Input tensor.
        """
        msg = (
            "This is a fake forward function. It must be masked on initialization by a _forward"
            " implementation."
        )
        raise NotImplementedError(f"Conv2DPycuda forward: {msg}")

    def backward(self, dy: TensorArray) -> TensorArray:
        """
        Performs the backward pass. Must be implemented by subclasses.

        Args:
            dy: Gradient of the output.
        """
        msg = (
            "This is a fake backward function. It must be masked on initialization by a _backward"
            " implementation."
        )
        raise NotImplementedError(f"Conv2DPycuda backward: {msg}")

    def _export_weights_dw(self, key: str) -> Any:
        """
        Exports weights or gradients of weights. Must be implemented by subclasses.

        Args:
            key: The parameter key to export.
        """
        # NOTE: Every variant must implement their version of this method.
        # super()._export_prop(key)
        msg = "This is a fake function. It must be overrided by the child classes."
        raise NotImplementedError(f"Conv2DPycuda export: {msg}")

    def _export_biases_db(self, key: str) -> Any:
        """
        Exports biases or gradients of biases to CPU.

        Args:
            key: The parameter key to export.
        """
        value = getattr(self, key)
        gpu_ary = value.ary
        cpu_ary = gpu_ary.get()

        match self.model.tensor_format:
            case TensorFormat.NHWC:
                return np.asarray(cpu_ary, dtype=np.float64, order="C", copy=True)
            case TensorFormat.NCHW:
                return np.asarray(cpu_ary, dtype=np.float64, order="C", copy=True)
            case tensor_format:
                raise TypeError(f"Unsupported tensor format ({tensor_format})")

    def _export_prop(self, key: str) -> Any:
        """
        Routes property export requests to the appropriate handler.

        Args:
            key: The parameter key to export.
        """
        match key:
            case Parameters.WEIGHTS | Parameters.DW:
                return self._export_weights_dw(key)
            case Parameters.BIASES | Parameters.DB:
                return self._export_biases_db(key)
            case _:
                return super()._export_prop(key)

    def _import_biases_db(self, key: str, value: Any) -> None:
        """
        Imports biases or gradients of biases from CPU.

        Args:
            key: The parameter key to import.
            value: The data to import.
        """
        attribute = getattr(self, key)

        match self.model.tensor_format:
            case TensorFormat.NHWC:
                cpu_ary = value
                attribute.set(cpu_ary)
                return
            case TensorFormat.NCHW:
                cpu_ary = value
                attribute.set(cpu_ary)
                return
            case tensor_format:
                raise TypeError(f"Unsupported tensor format ({tensor_format})")

    def _import_weights_dw(self, key: str, value: Any) -> None:
        """
        Imports weights or gradients of weights. Must be implemented by subclasses.

        Args:
            key: The parameter key to import.
            value: The data to import.
        """
        # NOTE: Every variant must implement their version of this method.
        # super()._export_prop(key)
        msg = "This is a fake function. It must be overrided by the child classes"
        raise NotImplementedError(f"Conv2DPycuda forward: {msg}")

    def _import_prop(self, key: str, value) -> None:
        """
        Routes property import requests to the appropriate handler.

        Args:
            key: The parameter key to import.
            value: The data to import.
        """
        match key:
            case Parameters.WEIGHTS | Parameters.DW:
                return self._import_weights_dw(key, value)
            case Parameters.BIASES | Parameters.DB:
                return self._import_biases_db(key, value)
            case _:
                return super()._import_prop(key, value)
