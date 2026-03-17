import logging
logger = logging.getLogger(__name__)

from pydtnn.backends.numpy.layers.conv_2d_depthwise import Conv2DDepthwiseNumpy
from pydtnn.backends.cython.utils.depthwise_conv_nchw_cython import depthwise_conv_backward_nchw_cython, depthwise_conv_nchw_cython
from pydtnn.backends.cython.utils.depthwise_conv_nhwc_cython import depthwise_conv_backward_nhwc_cython, depthwise_conv_nhwc_cython
from pydtnn.tracers.events import PYDTNN_OPS_EVENT, PYDTNN_OPS_EVENTS, PYDTNN_EVENT_FINISHED, PYDTNN_OPS_EVENT_enum

from pydtnn.libs import numpy as np
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    import numpy as np


class Conv2DDepthwiseCython(Conv2DDepthwiseNumpy):

    def _conv_fwd_nhwc(self, x: np.ndarray, y: np.ndarray) -> None:
        depthwise_conv_nhwc_cython(x, self.weights, y, self.ho, self.wo,
                                   self.hpadding, self.wpadding,
                                   self.hstride, self.wstride, self.hdilation, self.wdilation)
    # ----

    def _conv_fwd_nchw(self, x: np.ndarray, y: np.ndarray) -> None:
        depthwise_conv_nchw_cython(x, self.weights, y, self.ho, self.wo,
                                   self.hpadding, self.wpadding,
                                   self.hstride, self.wstride, self.hdilation, self.wdilation)
    # ----

    def _conv_bwd_nhwc(self, dx: np.ndarray, dy: np.ndarray) -> None:
        depthwise_conv_backward_nhwc_cython(dy, self.x, self.weights,
                                            dx, self.dw,
                                            self.hpadding, self.wpadding,
                                            self.hstride, self.wstride,
                                            self.hdilation, self.wdilation)
    # ----

    def _conv_bwd_nchw(self, dx: np.ndarray, dy: np.ndarray) -> None:
        depthwise_conv_backward_nchw_cython(dy, self.x, self.weights,
                                            dx, self.dw,
                                            self.hpadding, self.wpadding,
                                            self.hstride, self.wstride,
                                            self.hdilation, self.wdilation)
    # ----
