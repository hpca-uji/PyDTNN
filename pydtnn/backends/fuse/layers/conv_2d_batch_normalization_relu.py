"""Fused layer implementation for Conv2D, BatchNormalization, and ReLU operations."""

import logging
from typing import TYPE_CHECKING, Any

from pydtnn.activations.relu import Relu
from pydtnn.backends.fuse.layers.abstract.layer import LayerFuse
from pydtnn.backends.numpy.layers.abstract.conv_2d_standard import AbstractConv2DStandardNumpy
from pydtnn.backends.numpy.layers.batch_normalization import BatchNormalizationNumpy
from pydtnn.layers.batch_normalization import BatchNormalization
from pydtnn.layers.conv_2d import Conv2D
from pydtnn.libs import numpy as np
from pydtnn.tracers.events import (PYDTNN_EVENT_FINISHED, PYDTNN_OPS_EVENT,
                                   PYDTNN_OPS_EVENTS, OpsEventEnum)
from pydtnn.utils.constants import ArrayShape, Parameters

__all__ = ("Conv2DBatchNormalizationReluFuse",)


logger = logging.getLogger(__name__)


if TYPE_CHECKING:
    import numpy as np  # noqa: F811 (override typing)


class Conv2DBatchNormalizationReluFuse(
    LayerFuse, Conv2D[np.ndarray], BatchNormalization[np.ndarray], AbstractConv2DStandardNumpy
):
    """Numpy backend implementation for fused Conv2D, BatchNormalization, and ReLU."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Unpack fused parents"""
        super().__init__(*args, **kwargs)
        self.conv_2d: Conv2D[np.ndarray] = self.parents[0]
        self.batch_normalization: BatchNormalization[np.ndarray] = self.parents[1]
        self.relu: Relu[np.ndarray] = self.parents[2]

    @property
    def _ary_prop(self) -> set[str]:
        """Returns the set of array properties required for this fused layer."""
        return {Parameters.RUNNING_MEAN, Parameters.RUNNING_VAR, *super()._ary_prop}

    def _model_init(self, prev_shape: ArrayShape, x: np.ndarray) -> None:
        """Initializes layer parameters and selects the appropriate forward pass implementation."""
        super()._model_init(prev_shape, x)

        self.inv_std = BatchNormalizationNumpy.get_inv_std(
            self.running_var, self.epsilon, self.model.dtype
        )
        self.memory_used += self.inv_std.nbytes

        self.forward = {
            "_forward_cw_nchw": self._forward_nchw_cw,
            "_forward_cg_nchw": self._forward_nchw_cg,
            "_forward_cg_nhwc": self._forward_nhwc_cg,
        }[self.forward.__name__]
        self.backward = self._backward

    def _forward_nchw_cw(self, x: np.ndarray) -> np.ndarray:
        """Version of the forward function that uses the convWinograd + BatchNorm + Relu"""
        self.model.tracer.emit_event(
            PYDTNN_OPS_EVENT, self.id * PYDTNN_OPS_EVENTS + OpsEventEnum.FORWARD_CONVGEMM
        )
        y: np.ndarray = self.cw.conv_winograd_nchw(
            self.conv_2d.weights,
            x,
            self.conv_2d.biases,
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
            gamma=self.batch_normalization.weights,
            beta=self.batch_normalization.biases,
        )
        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, PYDTNN_EVENT_FINISHED)

        return np.asarray(y, dtype=self.model.dtype, order="C")

    def _forward_nchw_cg(self, x: np.ndarray) -> np.ndarray:
        """Version of the forward function that uses the convGemm + BatchNorm + Relu"""
        self.model.tracer.emit_event(
            PYDTNN_OPS_EVENT, self.id * PYDTNN_OPS_EVENTS + OpsEventEnum.FORWARD_CONVGEMM
        )
        res: np.ndarray = self.cg.conv_gemm_nchw(
            self.conv_2d.weights,
            x,
            vpadding=self.hpadding,
            hpadding=self.wpadding,
            vstride=self.hstride,
            hstride=self.wstride,
            vdilation=self.hdilation,
            hdilation=self.wdilation,
            biases=self.conv_2d.biases,
            bn_running_mean=self.running_mean,
            bn_inv_std=self.inv_std,
            bn_gamma=self.batch_normalization.weights,
            bn_beta=self.batch_normalization.biases,
            relu=True,
        )
        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, PYDTNN_EVENT_FINISHED)
        return np.asarray(res, dtype=self.model.dtype, order="C")

    def _forward_nhwc_cg(self, x: np.ndarray) -> np.ndarray:
        """Version of the forward function that uses the convGemm + BatchNorm + Relu"""
        self.model.tracer.emit_event(
            PYDTNN_OPS_EVENT, self.id * PYDTNN_OPS_EVENTS + OpsEventEnum.FORWARD_CONVGEMM
        )
        res: np.ndarray = self.cg.conv_gemm_nhwc(
            self.conv_2d.weights,
            x,
            vpadding=self.hpadding,
            hpadding=self.wpadding,
            vstride=self.hstride,
            hstride=self.wstride,
            vdilation=self.hdilation,
            hdilation=self.wdilation,
            biases=self.conv_2d.biases,
            bn_running_mean=self.running_mean,
            bn_inv_std=self.inv_std,
            bn_gamma=self.batch_normalization.weights,
            bn_beta=self.batch_normalization.biases,
            relu=True,
        )
        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, PYDTNN_EVENT_FINISHED)
        return np.asarray(res, dtype=self.model.dtype, order="C")

    def _backward(self, dy: np.ndarray) -> np.ndarray:
        """Raises NotImplementedError as backward pass is not supported for this fused layer."""
        raise NotImplementedError("Use a real backwards variant!")


# NOTE: select compatibility
Conv2DBatchNormalizationRelu = Conv2DBatchNormalizationReluFuse
