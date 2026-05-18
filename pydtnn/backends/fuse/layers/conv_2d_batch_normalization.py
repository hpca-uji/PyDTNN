"""
Module for fused 2D Convolution and Batch Normalization layers.
"""

import logging
from typing import TYPE_CHECKING

from pydtnn.backends.numpy.layers.abstract.conv_2d_standard import AbstractConv2DStandardNumpy
from pydtnn.backends.numpy.layers.batch_normalization import BatchNormalizationNumpy
from pydtnn.layers.conv_2d_batch_normalization import Conv2DBatchNormalization
from pydtnn.libs import numpy as np
from pydtnn.tracers.events import PYDTNN_EVENT_FINISHED, PYDTNN_OPS_EVENT, PYDTNN_OPS_EVENTS, PYDTNN_OPS_EVENT_enum
from pydtnn.utils.constants import ArrayShape, Parameters

__all__ = ("Conv2DBatchNormalizationFuse",)


logger = logging.getLogger(__name__)


if TYPE_CHECKING:
    import numpy as np


class Conv2DBatchNormalizationFuse(Conv2DBatchNormalization[np.ndarray], AbstractConv2DStandardNumpy):
    """
    Numpy-based implementation of fused 2D Convolution and Batch Normalization.
    """

    @property
    def _ary_prop(self) -> set[str]:
        """
        Returns the set of array properties required for this layer.
        """
        return {Parameters.RUNNING_MEAN, Parameters.RUNNING_VAR, *super()._ary_prop}

    # NOTE: The "__init__" method is being made (more or less) in Model (in _apply_layer_fusion) and in FusedLayerMixIn.

    def _model_init(self, prev_shape: ArrayShape, x: np.ndarray | None = None) -> None:
        """
        Initializes the layer parameters and selects the appropriate forward pass implementation.
        """
        super()._model_init(prev_shape, x)

        self.inv_std = BatchNormalizationNumpy.get_inv_std(self.running_var, self.epsilon, self.model.dtype)
        self.memory_used += self.inv_std.nbytes

        self.forward = {"_forward_cw_nchw": self._forward_nchw_cw, "_forward_cg_nchw": self._forward_nchw_cg, "_forward_cg_nhwc": self._forward_nhwc_cg}[self.forward.__name__]
        self.backward = self._backward

    def _forward_nchw_cw(self, x: np.ndarray) -> np.ndarray:
        """
        Performs the forward pass using Winograd convolution fused with Batch Normalization (NCHW).
        """
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
            relu=False,
            bn=True,
            running_mean=self.running_mean,
            inv_std=self.inv_std,
            gamma=self.gamma,
            beta=self.beta,
        )
        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, PYDTNN_EVENT_FINISHED)

        return np.asarray(y, dtype=self.model.dtype, order="C")

    def _forward_nchw_cg(self, x: np.ndarray) -> np.ndarray:
        """
        Performs the forward pass using GEMM-based convolution fused with Batch Normalization (NCHW).
        """
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
            relu=False,
        )
        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, PYDTNN_EVENT_FINISHED)
        return np.asarray(res, dtype=self.model.dtype, order="C")

    def _forward_nhwc_cg(self, x: np.ndarray) -> np.ndarray:
        """
        Performs the forward pass using GEMM-based convolution fused with Batch Normalization (NHWC).
        """
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
            relu=False,
        )
        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, PYDTNN_EVENT_FINISHED)
        return np.asarray(res, dtype=self.model.dtype, order="C")

    def _backward(self, dy: np.ndarray) -> np.ndarray:
        """
        Placeholder for the backward pass.
        """
        raise NotImplementedError("Use a real backward variant!")
