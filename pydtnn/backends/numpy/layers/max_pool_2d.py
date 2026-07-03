"""NumPy backend implementation of the 2D Max Pooling layer."""

import logging
from typing import TYPE_CHECKING, Any

from pydtnn.backends.numpy.layers.abstract.pool_2d_layer import AbstractPool2DLayerNumpy
from pydtnn.layers.max_pool_2d import MaxPool2D
from pydtnn.libs import numpy as np
from pydtnn.tracers.events import (PYDTNN_EVENT_FINISHED, PYDTNN_OPS_EVENT,
                                   PYDTNN_OPS_EVENTS, OpsEventEnum)
from pydtnn.utils.constants import ArrayShape

__all__ = ("MaxPool2DNumpy",)


logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    import numpy as np  # noqa: F811 (override typing)


class MaxPool2DNumpy(MaxPool2D[np.ndarray], AbstractPool2DLayerNumpy):
    """NumPy-based 2D Max Pooling layer implementation."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initialize the MaxPool2DNumpy layer."""
        super().__init__(*args, **kwargs)
        # The following attribute will be intialized later.
        self.idx_max: np.ndarray[tuple[int, int, int, int], np.int32] = None  # type: ignore
        self.y: np.ndarray  # NOTE: Defined and initalized in AbstractPool2DLayerNumpy's init and initialize, respectively

    def _model_init(self, prev_shape: ArrayShape, x: np.ndarray) -> None:
        """Initialize model parameters and allocate memory for indices."""
        super()._model_init(prev_shape, x)
        self.minval = (
            int(np.iinfo(self.model.dtype).min)
            if np.issubdtype(self.model.dtype, np.integer)
            else float(np.finfo(self.model.dtype).min)
        )
        idx_max_shape = self.model.encode_shape((self.model.batch_size, self.co, self.ho, self.wo))

        # NOTE: This attribute only stores data, its value before the operation
        # doesn't matter; it's initalized due avoid warnings in
        # "LayerAndActivationBase.export".
        self._idx_max: np.ndarray[tuple[int, int, int, int], np.int32] = np.zeros(idx_max_shape, dtype=np.int32)  # type: ignore
        self.memory_used += self._idx_max.nbytes

    def _fwd_max_pool_nhwc(self, x: np.ndarray, y: np.ndarray) -> None:
        """Perform forward pass for NHWC layout."""
        for nn in range(x.shape[0]):
            for xx in range(self.ho):
                for yy in range(self.wo):
                    for cc in range(self.ci):
                        maxval = self.minval
                        idx_maxval = 0
                        for ii in range(self.kh):
                            x_x = self.hstride * xx + self.hdilation * ii - self.hpadding
                            if 0 <= x_x < self.hi:
                                for jj in range(self.kw):
                                    x_y = self.wstride * yy + self.wdilation * jj - self.wpadding
                                    if 0 <= x_y < self.wi:
                                        val = x[nn, x_x, x_y, cc]
                                        if val > maxval:
                                            maxval = val
                                            idx_maxval = ii * self.kw + jj
                        y[nn, xx, yy, cc] = maxval
                        self.idx_max[nn, xx, yy, cc] = idx_maxval

    def _fwd_max_pool_nchw(self, x: np.ndarray, y: np.ndarray) -> None:
        """Perform forward pass for NCHW layout."""
        for nn in range(x.shape[0]):
            for cc in range(self.ci):
                for xx in range(self.ho):
                    for yy in range(self.wo):
                        maxval = self.minval
                        idx_maxval = 0
                        for ii in range(self.kh):
                            x_x = self.hstride * xx + self.hdilation * ii - self.hpadding
                            if 0 <= x_x < self.hi:
                                for jj in range(self.kw):
                                    x_y = self.wstride * yy + self.wdilation * jj - self.wpadding
                                    if 0 <= x_y < self.wi:
                                        val = x[nn, cc, x_x, x_y]
                                        if val > maxval:
                                            maxval = val
                                            idx_maxval = ii * self.kw + jj
                        y[nn, cc, xx, yy] = maxval
                        self.idx_max[nn, cc, xx, yy] = idx_maxval

    def _bwd_max_pool_nhwc(self, dx: np.ndarray, dy: np.ndarray) -> None:
        """Perform backward pass for NHWC layout."""
        for nn in range(dy.shape[0]):
            for xx in range(self.ho):
                for yy in range(self.wo):
                    for cc in range(self.ci):
                        idx_maxval = self.idx_max[nn, xx, yy, cc]
                        ii = idx_maxval // self.kh
                        jj = idx_maxval % self.kw
                        x_x = self.hstride * xx + self.hdilation * ii - self.hpadding
                        x_y = self.wstride * yy + self.wdilation * jj - self.wpadding
                        if 0 <= x_x < self.ho and 0 <= x_y < self.wo:
                            dx[nn, x_x, x_y, cc] += dy[nn, xx, yy, cc]

    def _bwd_max_pool_nchw(self, dx: np.ndarray, dy: np.ndarray) -> None:
        """Perform backward pass for NCHW layout."""
        for nn in range(dy.shape[0]):
            for cc in range(self.ci):
                for xx in range(self.ho):
                    for yy in range(self.wo):
                        idx_maxval = self.idx_max[nn, cc, xx, yy]
                        ii = idx_maxval // self.kh
                        jj = idx_maxval % self.kw
                        x_x = self.hstride * xx + self.hdilation * ii - self.hpadding
                        x_y = self.wstride * yy + self.wdilation * jj - self.wpadding
                        if 0 <= x_x < self.ho and 0 <= x_y < self.wo:
                            dx[nn, cc, x_x, x_y] += dy[nn, cc, xx, yy]

    def _forward_nhwc(self, x: np.ndarray) -> np.ndarray:
        """Execute forward pass in NHWC format."""
        # y:np.ndarray = self.y[:x.shape[0], :]
        y = self.get_y(x.shape[0])
        self.idx_max = self._idx_max[: x.shape[0], :]

        self.model.tracer.emit_event(
            PYDTNN_OPS_EVENT, self.id * PYDTNN_OPS_EVENTS + OpsEventEnum.FORWARD_IM2COL
        )
        self._fwd_max_pool_nhwc(x, y)
        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, PYDTNN_EVENT_FINISHED)
        return np.asarray(y, dtype=self.model.dtype, order="C")

    def _forward_nchw(self, x: np.ndarray) -> np.ndarray:
        """Execute forward pass in NCHW format."""
        # y:np.ndarray = self.y[:x.shape[0], :]
        y = self.get_y(x.shape[0])
        self.idx_max = self._idx_max[: x.shape[0], :]

        self.model.tracer.emit_event(
            PYDTNN_OPS_EVENT, self.id * PYDTNN_OPS_EVENTS + OpsEventEnum.FORWARD_IM2COL
        )
        self._fwd_max_pool_nchw(x, y)
        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, PYDTNN_EVENT_FINISHED)
        return np.asarray(y, dtype=self.model.dtype, order="C")

    def _backward_nhwc(self, dy: np.ndarray) -> np.ndarray:
        """Execute backward pass in NHWC format."""
        # dx:np.ndarray = self.dx[ :dy.shape[0], :]
        dx = self.get_dx(dy.shape[0])
        dx.fill(0)

        self.model.tracer.emit_event(
            PYDTNN_OPS_EVENT, self.id * PYDTNN_OPS_EVENTS + OpsEventEnum.COMP_DX_COL2IM
        )
        self._bwd_max_pool_nhwc(dx, dy)
        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, PYDTNN_EVENT_FINISHED)
        return np.asarray(dx, dtype=self.model.dtype, order="C")

    def _backward_nchw(self, dy: np.ndarray) -> np.ndarray:
        """Execute backward pass in NCHW format."""
        # dx:np.ndarray = self.dx[ :dy.shape[0], :]
        dx = self.get_dx(dy.shape[0])
        dx.fill(0)

        self.model.tracer.emit_event(
            PYDTNN_OPS_EVENT, self.id * PYDTNN_OPS_EVENTS + OpsEventEnum.COMP_DX_COL2IM
        )
        self._bwd_max_pool_nchw(dx, dy)
        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, PYDTNN_EVENT_FINISHED)
        return np.asarray(dx, dtype=self.model.dtype, order="C")

    # TEST

    def max_pool(
        self, x: np.ndarray, y: np.ndarray, idx_maxval: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Vectorized max pooling operation for testing purposes."""
        x = np.pad(
            x,
            ((0, 0), (0, 0), (self.hpadding, self.hpadding), (self.wpadding, self.wpadding)),
            mode="constant",
        )
        for kh in range(self.kh):
            for kw in range(self.kw):
                h_start = kh * self.hdilation
                w_start = kw * self.wdilation
                h_end = h_start + self.hstride * self.ho
                w_end = w_start + self.wstride * self.wo

                _x = x[:, :, h_start: h_end: self.hstride, w_start: w_end: self.wstride]
                max_val: np.ndarray = np.max(_x, axis=(2, 3))
                _idx_maxval: np.ndarray = np.argmax(np.argmax(_x, axis=3), axis=2)

                # y[:, :, h_start:h_end:self.vstride, w_start:w_end:self.hstride] = max_val[:, :]
                # idx_maxval[:, :, h_start:h_end:self.vstride, w_start:w_end:self.hstride] = _idx_maxval[:, :]

                for i in range(h_start, h_end // self.hstride):
                    for j in range(w_start, w_end // self.wstride):
                        y[:, :, i, j] = max_val[:, :]
                        idx_maxval[:, :, i, j] = _idx_maxval[:, :]
        return (y, idx_maxval)
