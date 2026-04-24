from typing import TYPE_CHECKING
from pydtnn.libs import numpy as np
from pydtnn.backends.cython.utils.adaptive_avg_pooling_nchw_cython import adaptive_avg_pooling_bwd_nchw_cython
from pydtnn.backends.cython.utils.adaptive_avg_pooling_nhwc_cython import adaptive_avg_pooling_bwd_nhwc_cython
from pydtnn.backends.cython.utils.adaptive_avg_pooling_nchw_cython import adaptive_avg_pooling_fwd_nchw_cython
from pydtnn.backends.cython.utils.adaptive_avg_pooling_nhwc_cython import adaptive_avg_pooling_fwd_nhwc_cython
from pydtnn.backends.numpy.layers.adaptive_average_pool_2d import AdaptiveAveragePool2DNumpy
from pydtnn.backends.cython.layers.abstract.pool_2d_layer import AbstractPool2DLayerCython
import logging
logger = logging.getLogger(__name__)


if TYPE_CHECKING:
    import numpy as np


class AdaptiveAveragePool2DCython(AdaptiveAveragePool2DNumpy, AbstractPool2DLayerCython):
    def _fwd_nhwc(self, x: np.ndarray, y: np.ndarray) -> None:
        adaptive_avg_pooling_fwd_nhwc_cython(x, y)  # type: ignore
    # ----

    def _fwd_nchw(self, x: np.ndarray, y: np.ndarray) -> None:
        adaptive_avg_pooling_fwd_nchw_cython(x, y)  # type: ignore
    # ----

    def _bwd_nhwc(self, dx: np.ndarray, dy: np.ndarray) -> None:
        adaptive_avg_pooling_bwd_nhwc_cython(dx, dy)  # type: ignore
    # ----

    def _bwd_nchw(self, dx: np.ndarray, dy: np.ndarray) -> None:
        adaptive_avg_pooling_bwd_nchw_cython(dx, dy)  # type: ignore
    # ----
