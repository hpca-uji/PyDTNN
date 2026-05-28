"""
Fused Batch Normalization and ReLU layer implementation for PyDTNN.
"""

import logging
from typing import TYPE_CHECKING

from pydtnn.backends.fuse.layers.abstract.layer import LayerFuse
from pydtnn.backends.fuse.utils.bn_inference_cython import bn_relu_inference_cython
from pydtnn.backends.numpy.layers.batch_normalization import BatchNormalizationNumpy
from pydtnn.libs import numpy as np
from pydtnn.utils.constants import ArrayShape
from pydtnn.utils.tensor import TensorFormat, format_transpose

__all__ = ("BatchNormalizationReluFuse",)


logger = logging.getLogger(__name__)


if TYPE_CHECKING:
    import numpy as np


class BatchNormalizationReluFuse(LayerFuse, BatchNormalizationNumpy):
    """
    Numpy-based implementation of fused Batch Normalization and ReLU for inference.
    """

    # NOTE: The "__init__" method is being made (more or less) in Model (in
    # _apply_layer_fusion) and in FusedLayerMixIn.

    def _model_init(self, prev_shape: ArrayShape, x: np.ndarray | None = None):
        """
        Initializes layer parameters and precomputes inverse standard deviation.
        """
        super()._model_init(prev_shape, x)

        self.inv_std = BatchNormalizationNumpy.get_inv_std(
            self.running_var, self.epsilon, self.model.dtype
        )

        # NOTE: This attribute only stores data, its value before the operation
        # doesn't matter; it's initalized due avoid warnings in
        # "LayerAndActivationBase.export".
        self.y: np.ndarray = np.zeros(
            shape=(self.model.batch_size, *self.shape), dtype=self.model.dtype
        )
        self.forward = self._forward
        self.backward = self._backward

        self.memory_used += self.y.nbytes + self.inv_std.nbytes

    def _forward(self, x: np.ndarray) -> np.ndarray:
        """
        Performs the fused forward pass using Cython-optimized BN + ReLU operations.
        """

        n = x.shape[0]
        if self.spatial:
            x = format_transpose(x, self.model.tensor_format, TensorFormat.NHWC)
            x = x.reshape((-1, self.ci), copy=False)

        y: np.ndarray = self.y[:n, :]
        bn_relu_inference_cython(
            x,
            y.reshape((-1, self.ci), copy=False),
            self.running_mean,
            self.inv_std,
            self.gamma,
            self.beta,
        )  # type: ignore (it's fine)

        if self.spatial:
            y = y.reshape((n, self.hi, self.wi, self.ci))
            y = format_transpose(y, TensorFormat.NHWC, self.model.tensor_format)

        return np.asarray(y, dtype=self.model.dtype, order="C")

    def _backward(self, dy: np.ndarray) -> np.ndarray:
        """
        Raises NotImplementedError as this layer is intended for inference only.
        """
        raise NotImplementedError("Use a real backwards variant!")


# NOTE: select compatibility
BatchNormalizationRelu = BatchNormalizationReluFuse
