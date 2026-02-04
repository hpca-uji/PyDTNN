from pydtnn.backends.cpu.layers.adaptive_average_pool_2d import AdaptiveAveragePool2DCPU
from pydtnn.backends.cython.utils.adaptive_avg_pooling_nchw_cython import adaptive_avg_pooling_bwd_nchw_cython, adaptive_avg_pooling_fwd_nchw_cython
from pydtnn.backends.cython.utils.adaptive_avg_pooling_nhwc_cython import adaptive_avg_pooling_bwd_nhwc_cython, adaptive_avg_pooling_fwd_nhwc_cython

# Imports for the methods from AveragePool2DCPU
from pydtnn.tracers.events import PYDTNN_OPS_EVENT, PYDTNN_OPS_EVENTS, PYDTNN_EVENT_FINISHED, PYDTNN_OPS_EVENT_enum
from pydtnn.libs import numpy as np
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    import numpy as np


class AdaptiveAveragePool2DCYTHON(AdaptiveAveragePool2DCPU):
    # The backend is almost the same as a AveragePool2D layer.

    def _forward_nhwc(self, x: np.ndarray) -> np.ndarray:
        y: np.ndarray = self.y[:x.shape[0], :]
        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, self.id * PYDTNN_OPS_EVENTS + PYDTNN_OPS_EVENT_enum.FORWARD_ADP_AVG_POOL)
        adaptive_avg_pooling_fwd_nhwc_cython(x, y)
        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, PYDTNN_EVENT_FINISHED)
        return np.asarray(y, dtype=self.model.dtype)

    def _forward_nchw(self, x: np.ndarray) -> np.ndarray:
        y: np.ndarray = self.y[:x.shape[0], :]
        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, self.id * PYDTNN_OPS_EVENTS + PYDTNN_OPS_EVENT_enum.FORWARD_ADP_AVG_POOL)
        adaptive_avg_pooling_fwd_nchw_cython(x, y)
        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, PYDTNN_EVENT_FINISHED)
        return np.asarray(y, dtype=self.model.dtype)

    def _backward_nhwc(self, dy: np.ndarray) -> np.ndarray:
        dx: np.ndarray = self.dx[:dy.shape[0]]
        dx.fill(0)
        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, self.id * PYDTNN_OPS_EVENTS + PYDTNN_OPS_EVENT_enum.BACKWARD_ADP_AVG_POOL)
        adaptive_avg_pooling_bwd_nhwc_cython(dy, dx)
        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, PYDTNN_EVENT_FINISHED)
        return np.asarray(dx, dtype=self.model.dtype)

    def _backward_nchw(self, dy: np.ndarray) -> np.ndarray:
        dx: np.ndarray = self.dx[:dy.shape[0]]
        dx.fill(0)
        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, self.id * PYDTNN_OPS_EVENTS + PYDTNN_OPS_EVENT_enum.BACKWARD_ADP_AVG_POOL)
        adaptive_avg_pooling_bwd_nchw_cython(dy, dx)
        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, PYDTNN_EVENT_FINISHED)
        return np.asarray(dx, dtype=self.model.dtype)
