from pydtnn.cython.bn_inference_cython import bn_relu_inference_cython
from pydtnn.backends.cpu.layers.layer_cpu import LayerCPU
from pydtnn.layers.batch_normalization_relu import BatchNormalizationRelu
from pydtnn.model import Model
from pydtnn.utils.tensor import TensorFormat
from pydtnn.utils.best_transpose_0231 import best_transpose_0231
from pydtnn.utils.best_transpose_0312 import best_transpose_0312

import numpy as np


class BatchNormalizationReluCPU(LayerCPU, BatchNormalizationRelu[np.ndarray]):

    def initialize(self, prev_shape, x=None):
        super().initialize(prev_shape, x)
        self.y: np.ndarray = np.empty(shape=(self.model.batch_size, *self.shape), dtype=self.model.dtype, order="C")

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Version of the forward function that uses the BN + Relu"""

        if self.model.mode is Model.Mode.TRAIN:
            raise SystemExit("Sorry, fused layers cannot be used in training mode!")

        if self.spatial:
            if self.model.tensor_format is TensorFormat.NCHW:
                x = best_transpose_0231(x)
            x = x.reshape((-1, self.ci), copy=False, order="C")

        y: np.ndarray = self.y[: x.shape[0], :]
        bn_relu_inference_cython(x,
                                 y.reshape((-1, self.ci), copy=False, order="C"),
                                 self.running_mean,
                                 self.inv_std,
                                 self.gamma,
                                 self.beta)

        if self.spatial:
            y = y.reshape((-1, self.hi, self.wi, self.ci), copy=False)
            if self.model.tensor_format is TensorFormat.NCHW:
                y = best_transpose_0312(y)
        return np.asarray(y, dtype=self.model.dtype, order='C', copy=None)

    def backward(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError("Use a real backward variant!")
