"""
NumPy backend implementation for 2D Average Pooling layers.
"""
import logging
from typing import TYPE_CHECKING

from pydtnn.backends.numpy.layers.abstract.pool_2d_layer import AbstractPool2DLayerNumpy
from pydtnn.layers.average_pool_2d import AveragePool2D
from pydtnn.libs import numpy as np
from pydtnn.tracers.events import PYDTNN_EVENT_FINISHED, PYDTNN_OPS_EVENT, PYDTNN_OPS_EVENTS, PYDTNN_OPS_EVENT_enum

__all__ = ("AveragePool2DNumpy",)


logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    import numpy as np


class AveragePool2DNumpy(AveragePool2D[np.ndarray], AbstractPool2DLayerNumpy):
    """
    NumPy implementation of the 2D Average Pooling layer.
    """
    def _fwd_avg_pool_nchw(self, x: np.ndarray, y: np.ndarray) -> None:
        """
        Performs forward average pooling for NCHW input format.
        """
        for nn in range(x.shape[0]):
            for cc in range(self.ci):
                for xx in range(self.ho):
                    for yy in range(self.wo):
                        accum = 0.0
                        items = 0
                        # accum, items = 0, (kh * kw)
                        for ii in range(self.kh):
                            x_x = self.hstride * xx + self.hdilation * ii - self.hpadding
                            if 0 <= x_x < self.hi:
                                for jj in range(self.kw):
                                    x_y = self.wstride * yy + self.wdilation * jj - self.wpadding
                                    if 0 <= x_y < self.wi:
                                        accum += x[nn, cc, x_x, x_y]
                                        items += 1
                        y[nn, cc, xx, yy] = accum / items

    def _fwd_avg_pool_nhwc(self, x: np.ndarray, y: np.ndarray) -> None:
        """
        Performs forward average pooling for NHWC input format.
        """
        for nn in range(x.shape[0]):
            for xx in range(self.ho):
                for yy in range(self.wo):
                    for cc in range(self.ci):
                        accum = 0.0
                        items = 0
                        # accum, items = 0, (kh * kw)
                        for ii in range(self.kh):
                            x_x = self.hstride * xx + self.hdilation * ii - self.hpadding
                            if 0 <= x_x < self.hi:
                                for jj in range(self.kw):
                                    x_y = self.wstride * yy + self.wdilation * jj - self.wpadding
                                    if 0 <= x_y < self.wi:
                                        accum += x[nn, x_x, x_y, cc]
                                        items += 1
                        y[nn, xx, yy, cc] = accum / items

    def _bwd_avg_pool_nhwc(self, dx: np.ndarray, dy: np.ndarray) -> None:
        """
        Performs backward average pooling for NHWC input format.
        """
        for nn in range(dy.shape[0]):
            for xx in range(self.ho):
                for yy in range(self.wo):
                    for cc in range(self.ci):
                        items = 0
                        avgval = dy[nn, xx, yy, cc]
                        for ii in range(self.kh):
                            x_x = self.hstride * xx + self.hdilation * ii - self.hpadding
                            if 0 <= x_x < self.hi:
                                for jj in range(self.kw):
                                    x_y = self.wstride * yy + self.wdilation * jj - self.wpadding
                                    if 0 <= x_y < self.wi:
                                        items = items + 1
                        avgval /= items
                        # avgval = dy[nn, xx, yy, cc] // (kh * kw)
                        for ii in range(self.kh):
                            x_x = self.hstride * xx + self.hdilation * ii - self.hpadding
                            if 0 <= x_x < self.hi:
                                for jj in range(self.kw):
                                    x_y = self.wstride * yy + self.wdilation * jj - self.wpadding
                                    if 0 <= x_y < self.wi:
                                        dx[nn, x_x, x_y, cc] += avgval

    def _bwd_avg_pool_nchw(self, dx: np.ndarray, dy: np.ndarray) -> None:
        """
        Performs backward average pooling for NCHW input format.
        """
        for nn in range(dy.shape[0]):
            for cc in range(self.ci):
                for xx in range(self.ho):
                    for yy in range(self.wo):
                        items = 0
                        avgval = dy[nn, cc, xx, yy]
                        for ii in range(self.kh):
                            x_x = self.hstride * xx + self.hdilation * ii - self.hpadding
                            if 0 <= x_x < self.hi:
                                for jj in range(self.kw):
                                    x_y = self.wstride * yy + self.wdilation * jj - self.wpadding
                                    if 0 <= x_y < self.wi:
                                        items = items + 1
                        avgval /= items
                        for ii in range(self.kh):
                            x_x = self.hstride * xx + self.hdilation * ii - self.hpadding
                            if 0 <= x_x < self.hi:
                                for jj in range(self.kw):
                                    x_y = self.wstride * yy + self.wdilation * jj - self.wpadding
                                    if 0 <= x_y < self.wi:
                                        dx[nn, cc, x_x, x_y] += avgval

    def _forward_nchw(self, x: np.ndarray) -> np.ndarray:
        """
        Executes forward pass for NCHW data.
        """
        # y:np.ndarray = self.y[:x.shape[0], :]
        y = self.get_y(x.shape[0])
        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, self.id * PYDTNN_OPS_EVENTS + PYDTNN_OPS_EVENT_enum.FORWARD_IM2COL)
        self._fwd_avg_pool_nchw(x, y)
        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, PYDTNN_EVENT_FINISHED)
        return np.asarray(y, dtype=self.model.dtype, order="C")

    def _forward_nhwc(self, x: np.ndarray) -> np.ndarray:
        """
        Executes forward pass for NHWC data.
        """
        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, self.id * PYDTNN_OPS_EVENTS + PYDTNN_OPS_EVENT_enum.FORWARD_IM2COL)
        # y:np.ndarray = self.y[:x.shape[0], :]
        y = self.get_y(x.shape[0])
        self._fwd_avg_pool_nhwc(x, y)
        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, PYDTNN_EVENT_FINISHED)
        return np.asarray(y, dtype=self.model.dtype, order="C")

    def _backward_nhwc(self, dy: np.ndarray) -> np.ndarray:
        """
        Executes backward pass for NHWC data.
        """
        # NOTE: It's necessary a new zero-initalized "dx" in every call since may be some values that are not re-set in the cython's function.
        # dx:np.ndarray = self.dx[ :dy.shape[0], :]
        dx = self.get_dx(dy.shape[0])
        dx.fill(0)
        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, self.id * PYDTNN_OPS_EVENTS + PYDTNN_OPS_EVENT_enum.COMP_DX_COL2IM)
        self._bwd_avg_pool_nhwc(dx, dy)
        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, PYDTNN_EVENT_FINISHED)
        return np.asarray(dx, dtype=self.model.dtype, order="C")

    def _backward_nchw(self, dy: np.ndarray) -> np.ndarray:
        """
        Executes backward pass for NCHW data.
        """
        # NOTE: It's necessary a new zero-initalized "dx" in every call since may be some values that are not re-set in the cython's function.
        # dx:np.ndarray = self.dx[ :dy.shape[0], :]
        dx = self.get_dx(dy.shape[0])
        dx.fill(0)
        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, self.id * PYDTNN_OPS_EVENTS + PYDTNN_OPS_EVENT_enum.COMP_DX_COL2IM)
        self._bwd_avg_pool_nchw(dx, dy)
        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, PYDTNN_EVENT_FINISHED)
        return np.asarray(dx, dtype=self.model.dtype, order="C")