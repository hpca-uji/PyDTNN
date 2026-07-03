"""Fused 2D Convolution and ReLU layer implementation."""

import logging
from typing import TYPE_CHECKING

from pydtnn.backends.fuse.layers.abstract.layer import LayerFuse
from pydtnn.backends.numpy.layers.abstract.conv_2d_standard import AbstractConv2DStandardNumpy
from pydtnn.layers.conv_2d import Conv2D
from pydtnn.libs import numpy as np
from pydtnn.tracers.events import (PYDTNN_EVENT_FINISHED, PYDTNN_OPS_EVENT,
                                   PYDTNN_OPS_EVENTS, OpsEventEnum)
from pydtnn.utils.constants import ArrayShape

__all__ = ("Conv2DReluFuse",)


logger = logging.getLogger(__name__)


if TYPE_CHECKING:
    import numpy as np  # noqa: F811 (override typing)


class Conv2DReluFuse(LayerFuse, Conv2D[np.ndarray], AbstractConv2DStandardNumpy):
    """Numpy-based implementation of a fused 2D Convolution and ReLU layer."""

    # NOTE: The "__init__" method is being made (more or less) in Model (in
    # _apply_layer_fusion) and in FusedLayerMixIn.

    def _model_init(self, prev_shape: ArrayShape, x: np.ndarray) -> None:
        """Initializes the layer model and maps forward/backward methods."""
        super()._model_init(prev_shape, x)
        self.forward = {
            "_forward_cg_nchw": self._forward_nchw_cg,
            "_forward_cg_nhwc": self._forward_nhwc_cg,
            "_forward_cw_nchw": self._forward_nchw_cw,
        }[self.forward.__name__]
        self.backward = self._backward

    def _forward_nchw_cg(self, x: np.ndarray) -> np.ndarray:
        """Version of the forward function that uses the convGemm + Relu"""

        self.model.tracer.emit_event(
            PYDTNN_OPS_EVENT, self.id * PYDTNN_OPS_EVENTS + OpsEventEnum.FORWARD_CONVGEMM
        )
        res: np.ndarray = self.cg.conv_gemm_nchw(
            self.weights,
            x,
            out=None,
            vpadding=self.hpadding,
            hpadding=self.wpadding,
            vstride=self.hstride,
            hstride=self.wstride,
            vdilation=self.hdilation,
            hdilation=self.wdilation,
            biases=self.biases,
            relu=True,
        )
        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, PYDTNN_EVENT_FINISHED)
        return np.asarray(res, dtype=self.model.dtype, order="C")

    def _forward_nhwc_cg(self, x: np.ndarray) -> np.ndarray:
        """Version of the forward function that uses the convGemm + Relu"""

        self.model.tracer.emit_event(
            PYDTNN_OPS_EVENT, self.id * PYDTNN_OPS_EVENTS + OpsEventEnum.FORWARD_CONVGEMM
        )
        res: np.ndarray = self.cg.conv_gemm_nhwc(
            self.weights,
            x,
            out=None,
            vpadding=self.hpadding,
            hpadding=self.wpadding,
            vstride=self.hstride,
            hstride=self.wstride,
            vdilation=self.hdilation,
            hdilation=self.wdilation,
            biases=self.biases,
            relu=True,
        )
        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, PYDTNN_EVENT_FINISHED)
        return np.asarray(res, dtype=self.model.dtype, order="C")

    def _forward_nchw_cw(self, x: np.ndarray) -> np.ndarray:
        """Version of the forward function that uses the convWinograd + Relu"""
        self.model.tracer.emit_event(
            PYDTNN_OPS_EVENT, self.id * PYDTNN_OPS_EVENTS + OpsEventEnum.FORWARD_CONVGEMM
        )
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
        )
        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, PYDTNN_EVENT_FINISHED)

        return np.asarray(y, dtype=self.model.dtype, order="C")

    def _backward(self, dy: np.ndarray) -> np.ndarray:
        """Placeholder for backward pass, currently raises NotImplementedError."""
        raise NotImplementedError("Use a real backwards variant!")


# NOTE: select compatibility
Conv2DRelu = Conv2DReluFuse
