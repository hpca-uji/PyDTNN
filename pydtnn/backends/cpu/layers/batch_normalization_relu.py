from pydtnn.backends.cpu.utils.bn_inference_cython import bn_relu_inference_cython
from pydtnn.backends.cpu.layers.batch_normalization import BatchNormalizationCPU
from pydtnn.layers.batch_normalization_relu import BatchNormalizationRelu
from pydtnn.utils.constants import ArrayShape

import numpy as np


class BatchNormalizationReluCPU(BatchNormalizationRelu[np.ndarray], BatchNormalizationCPU):

    # NOTE: The "__init__" method is being made (more or less) in Model (in _apply_layer_fusion) and in FusedLayerMixIn.

    def initialize(self, prev_shape: ArrayShape, x: np.ndarray | None = None):
        super().initialize(prev_shape, x)

        self.inv_std = BatchNormalizationCPU.get_inv_std(self.running_var, self.epsilon, self.model.dtype)

        # NOTE: This attribute only stores data, its value before the operation doesn't matter; it's initalized due avoid warnings in "LayerAndActivationBase.export".
        self.y: np.ndarray = np.zeros(shape=(self.model.batch_size, *self.shape), dtype=self.model.dtype, order="C")
        self.forward = self._forward
        self.backward = self._backward

    def _forward(self, x: np.ndarray) -> np.ndarray:
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
            y_shape = self.model.encode_shape((n, self.ci, self.hi, self.wi))
            y = y.reshape(y_shape, copy=False)

        return np.asarray(y, dtype=self.model.dtype, order='C', copy=None)

    def _backward(self, dy: np.ndarray) -> np.ndarray:
        raise NotImplementedError("Use a real backwards variant!")
