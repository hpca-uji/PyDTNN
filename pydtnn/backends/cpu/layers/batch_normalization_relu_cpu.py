from pydtnn.cython.bn_inference_cython import bn_relu_inference_cython
from pydtnn.backends.cpu.layers.batch_normalization_cpu import BatchNormalizationCPU
from pydtnn.layers.batch_normalization_relu import BatchNormalizationRelu
from pydtnn.utils.tensor import TensorFormat
import typing

import numpy as np


class BatchNormalizationReluCPU(BatchNormalizationCPU, BatchNormalizationRelu[np.ndarray]):

    @typing.override
    def post_initialize(self):
        pass

    def initialize(self, prev_shape, x=None):
        super().initialize(prev_shape, x)
        self.y: np.ndarray = np.empty(shape=(self.model.batch_size, *self.shape), dtype=self.model.dtype, order="C")

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Version of the forward function that uses the BN + Relu"""

        n = x.shape[0]
        if self.spatial:
            x = x.reshape((-1, self.ci), copy=False, order="C")

        y: np.ndarray = self.y[:n, :]
        bn_relu_inference_cython(x,
                                 y.reshape((-1, self.ci), copy=False, order="C"),
                                 self.running_mean,
                                 self.inv_std,
                                 self.gamma,
                                 self.beta)

        if self.spatial:
            match self.model.tensor_format:
                case TensorFormat.NCHW:
                    y = y.reshape((n, self.ci, self.hi, self.wi), copy=False)
                case TensorFormat.NHWC:
                    y = y.reshape((n, self.hi, self.wi, self.ci), copy=False)
                case _:
                    raise ValueError(f"{self.model.tensor_format} tensor format not supported. Tensor format supported: {list(self.model.tensor_format)}")

        return np.asarray(y, dtype=self.model.dtype, order='C', copy=None)

    def backward(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError("Use a real backward variant!")
