"""Fused layer implementation for Conv2D, BatchNormalization, and ReLU operations."""
import logging
from typing import TYPE_CHECKING

from pydtnn.backends.fuse.layers.layer import LayerFuse as FusedLayerMixIn
from pydtnn.backends.numpy.layers.abstract.conv_2d_standard import AbstractConv2DStandardNumpy
from pydtnn.backends.numpy.layers.batch_normalization import BatchNormalizationNumpy
from pydtnn.layers.batch_normalization import BatchNormalization
from pydtnn.layers.conv_2d import Conv2D
from pydtnn.libs import numpy as np
from pydtnn.tracers.events import PYDTNN_EVENT_FINISHED, PYDTNN_OPS_EVENT, PYDTNN_OPS_EVENTS, PYDTNN_OPS_EVENT_enum
from pydtnn.utils.constants import Array, ArrayShape, Parameters

__all__ = (
    "Conv2DBatchNormalizationRelu",
    "Conv2DBatchNormalizationReluFuse",
)


logger = logging.getLogger(__name__)


if TYPE_CHECKING:
    import numpy as np


class Conv2DBatchNormalizationRelu[T: Array](FusedLayerMixIn[T], Conv2D[T], BatchNormalization[T]):
    """Base class for fused Conv2D, BatchNormalization, and ReLU layers."""
    pass


class Conv2DBatchNormalizationReluFuse(Conv2DBatchNormalizationRelu[np.ndarray], AbstractConv2DStandardNumpy):
    """Numpy backend implementation for fused Conv2D, BatchNormalization, and ReLU."""
    @property
    def _ary_prop(self) -> set[str]:
        """Returns the set of array properties required for this fused layer."""
        return {Parameters.RUNNING_MEAN, Parameters.RUNNING_VAR, *super()._ary_prop}

    # NOTE: The "__init__" method is being made (more or less) in Model (in _apply_layer_fusion) and in FusedLayerMixIn.

    def _model_init(self, prev_shape: ArrayShape, x: np.ndarray | None = None) -> None:
        """Initializes layer parameters and selects the appropriate forward pass implementation."""
        super()._model_init(prev_shape, x)

        self.inv_std = BatchNormalizationNumpy.get_inv_std(self.running_var, self.epsilon, self.model.dtype)
        self.memory_used += self.inv_std.nbytes

        self.forward = {"_forward_cw_nchw": self._forward_nchw_cw, "_forward_cg_nchw": self._forward_nchw_cg, "_forward_cg_nhwc": self._forward_nhwc_cg}[self.forward.__name__]
        self.backward = self._backward

    def _forward_nchw_cw(self, x: np.ndarray) -> np.ndarray:
        """Version of the forward function that uses the convWinograd + BatchNorm + Relu"""
        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, self.id * PYDTNN_OPS_EVENTS + PYDTNN_OPS_EVENT_enum.FORWARD_CONVGEMM)
        y: np.ndarray = self.cw.conv_winograd_nchw(
            self.weights,
            x,
            self.biases,
            vpadding=self.hpadding,
            hpadding=self.wpadding,
            vstride=self.hstride,
            hstride=self.wstride,
            vdilation=self.hdilation,
            hdilation=self.wdilation,
            relu=True,
            bn=True,
            running_mean=self.running_mean,
            inv_std=self.inv_std,
            gamma=self.gamma,
            beta=self.beta,
        )
        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, PYDTNN_EVENT_FINISHED)

        return np.asarray(y, dtype=self.model.dtype, order="C")

    def _forward_nchw_cg(self, x: np.ndarray) -> np.ndarray:
        """Version of the forward function that uses the convGemm + BatchNorm + Relu"""
        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, self.id * PYDTNN_OPS_EVENTS + PYDTNN_OPS_EVENT_enum.FORWARD_CONVGEMM)
        res: np.ndarray = self.cg.conv_gemm_nchw(
            self.weights,
            x,
            vpadding=self.hpadding,
            hpadding=self.wpadding,
            vstride=self.hstride,
            hstride=self.wstride,
            vdilation=self.hdilation,
            hdilation=self.wdilation,
            biases=self.biases,
            bn_running_mean=self.running_mean,
            bn_inv_std=self.inv_std,
            bn_gamma=self.gamma,
            bn_beta=self.beta,
            relu=True,
        )
        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, PYDTNN_EVENT_FINISHED)
        return np.asarray(res, dtype=self.model.dtype, order="C")

    def _forward_nhwc_cg(self, x: np.ndarray) -> np.ndarray:
        """Version of the forward function that uses the convGemm + BatchNorm + Relu"""
        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, self.id * PYDTNN_OPS_EVENTS + PYDTNN_OPS_EVENT_enum.FORWARD_CONVGEMM)
        res: np.ndarray = self.cg.conv_gemm_nhwc(
            self.weights,
            x,
            vpadding=self.hpadding,
            hpadding=self.wpadding,
            vstride=self.hstride,
            hstride=self.wstride,
            vdilation=self.hdilation,
            hdilation=self.wdilation,
            biases=self.biases,
            bn_running_mean=self.running_mean,
            bn_inv_std=self.inv_std,
            bn_gamma=self.gamma,
            bn_beta=self.beta,
            relu=True,
        )
        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, PYDTNN_EVENT_FINISHED)
        return np.asarray(res, dtype=self.model.dtype, order="C")

    def _backward(self, dy: np.ndarray) -> np.ndarray:
        """Raises NotImplementedError as backward pass is not supported for this fused layer."""
        raise NotImplementedError("Use a real backwards variant!")