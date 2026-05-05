import logging
from typing import TYPE_CHECKING

from pydtnn.backends.cython.layers.abstract.pool_2d_layer import AbstractPool2DLayerCython
from pydtnn.backends.cython.utils.argmax_cython import argmax_cython
from pydtnn.backends.cython.utils.im2col_1ch_nchw_cython import col2im_1ch_nchw_cython, im2col_1ch_nchw_cython
from pydtnn.backends.cython.utils.im2row_1ch_nhwc_cython import im2row_1ch_nhwc_cython, row2im_1ch_nhwc_cython
from pydtnn.backends.cython.utils.max_pool_2d_nchw_cython import max_pool_2d_bwd_nchw_cython, max_pool_2d_fwd_nchw_cython
from pydtnn.backends.cython.utils.max_pool_2d_nhwc_cython import max_pool_2d_bwd_nhwc_cython, max_pool_2d_fwd_nhwc_cython
from pydtnn.backends.numpy.layers.max_pool_2d import MaxPool2DNumpy
from pydtnn.libs import numpy as np
from pydtnn.model import Model
from pydtnn.tracers.events import PYDTNN_EVENT_FINISHED, PYDTNN_OPS_EVENT, PYDTNN_OPS_EVENTS, PYDTNN_OPS_EVENT_enum

__all__ = (
    "MaxPool2DCython",
)


logger = logging.getLogger(__name__)


if TYPE_CHECKING:
    import numpy as np


class MaxPool2DCython(MaxPool2DNumpy, AbstractPool2DLayerCython):
    # CYTHON

    def _fwd_max_pool_nhwc(self, x: np.ndarray, y: np.ndarray) -> None:
        max_pool_2d_fwd_nhwc_cython(
            x,
            y,
            self.idx_max,  # type: ignore
            self.kh,
            self.kw,
            self.ho,
            self.wo,
            self.hpadding,
            self.wpadding,
            self.hstride,
            self.wstride,
            self.hdilation,
            self.wdilation,
            self.minval,
        )

    def _fwd_max_pool_nchw(self, x: np.ndarray, y: np.ndarray) -> None:
        max_pool_2d_fwd_nchw_cython(
            x,
            y,
            self.idx_max,  # type: ignore
            self.kh,
            self.kw,
            self.ho,
            self.wo,
            self.hpadding,
            self.wpadding,
            self.hstride,
            self.wstride,
            self.hdilation,
            self.wdilation,
            self.minval,
        )

    def _bwd_max_pool_nhwc(self, dx: np.ndarray, dy: np.ndarray) -> None:
        max_pool_2d_bwd_nhwc_cython(
            dy,
            self.idx_max,
            dx,  # type: ignore
            dy.shape[0],
            self.hi,
            self.wi,
            self.ci,
            self.kh,
            self.kw,
            self.ho,
            self.wo,
            self.hpadding,
            self.wpadding,
            self.hstride,
            self.wstride,
            self.hdilation,
            self.wdilation,
        )

    def _bwd_max_pool_nchw(self, dx: np.ndarray, dy: np.ndarray) -> None:
        max_pool_2d_bwd_nchw_cython(
            dy,
            self.idx_max,
            dx,  # type: ignore
            dy.shape[0],
            self.hi,
            self.wi,
            self.ci,
            self.kh,
            self.kw,
            self.ho,
            self.wo,
            self.hpadding,
            self.wpadding,
            self.hstride,
            self.wstride,
            self.hdilation,
            self.wdilation,
        )

    # I2C

    def _forward_nhwc_i2c(self, x: np.ndarray) -> np.ndarray:
        y: np.ndarray = np.zeros((x.shape[0],), dtype=self.model.dtype)
        amax: np.ndarray = np.zeros((x.shape[0],), dtype=np.int32)
        rng: np.ndarray = np.zeros((x.shape[0],), dtype=np.int32)
        x_rows: np.ndarray = np.zeros((x.shape[0] * self.ci * self.ho * self.wo, self.kh * self.kw), dtype=self.model.dtype)

        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, self.id * PYDTNN_OPS_EVENTS + PYDTNN_OPS_EVENT_enum.FORWARD_IM2COL)
        im2row_1ch_nhwc_cython(
            x,
            x_rows,  # type: ignore
            self.kh,
            self.kw,
            self.ho,
            self.wo,
            self.hpadding,
            self.wpadding,
            self.hstride,
            self.wstride,
            self.hdilation,
            self.wdilation,
        )
        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, PYDTNN_EVENT_FINISHED)
        idx_max: np.ndarray = argmax_cython(x_rows, y, amax, rng, axis=1)  # type: ignore

        idx_max: np.ndarray
        if self.model.mode is Model.Mode.TRAIN:
            self.idx_max = idx_max
        return y.reshape((-1, self.ho, self.wo, self.co))

    def _forward_nchw_i2c(self, x: np.ndarray) -> np.ndarray:
        n, c, _, _ = x.shape
        x_cols: np.ndarray = np.zeros((self.kh * self.kw, n * c * self.ho * self.wo), dtype=self.model.dtype)
        y: np.ndarray = np.zeros((n,), dtype=self.model.dtype)
        amax: np.ndarray = np.zeros((n,), dtype=np.int32)
        rng: np.ndarray = np.zeros((n,), dtype=np.int32)

        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, self.id * PYDTNN_OPS_EVENTS + PYDTNN_OPS_EVENT_enum.FORWARD_IM2COL)
        im2col_1ch_nchw_cython(
            x,
            x_cols,  # type: ignore
            self.kh,
            self.kw,
            self.ho,
            self.wo,
            self.hpadding,
            self.wpadding,
            self.hstride,
            self.wstride,
            self.hdilation,
            self.wdilation,
        )
        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, PYDTNN_EVENT_FINISHED)
        idx_max: np.ndarray = argmax_cython(x_cols, y, amax, rng, axis=0)  # type: ignore
        if self.model.mode is Model.Mode.TRAIN:
            self.idx_max = idx_max
        return y.reshape((-1, self.co, self.ho, self.wo))

    def _backward_nhwc_i2c(self, dy: np.ndarray) -> np.ndarray:
        dy_rows: np.ndarray = np.zeros((np.prod(dy.shape), self.kh * self.kw), dtype=self.model.dtype)
        dy_rows[self.idx_max] = dy.flatten()
        dx: np.ndarray = np.zeros_like(dy, dtype=self.model.dtype)
        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, self.id * PYDTNN_OPS_EVENTS + PYDTNN_OPS_EVENT_enum.COMP_DX_COL2IM)
        row2im_1ch_nhwc_cython(
            dy_rows,
            dx,  # type: ignore
            dy.shape[0],
            self.hi,
            self.wi,
            self.ci,
            self.kh,
            self.kw,
            self.ho,
            self.wo,
            self.hpadding,
            self.wpadding,
            self.hstride,
            self.wstride,
            self.hdilation,
            self.wdilation,
        )
        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, PYDTNN_EVENT_FINISHED)
        return dx.reshape((-1, self.hi, self.wi, self.ci))

    def _backward_nchw_i2c(self, dy: np.ndarray) -> np.ndarray:
        dy_cols: np.ndarray = np.zeros((self.kh * self.kw, np.prod(dy.shape)), dtype=self.model.dtype)
        dy_cols[self.idx_max] = dy.flatten(order="C").view(dtype=self.model.dtype)
        dx: np.ndarray = np.zeros((dy.shape[0], self.ci, self.hi, self.wi), dtype=self.model.dtype)

        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, self.id * PYDTNN_OPS_EVENTS + PYDTNN_OPS_EVENT_enum.COMP_DX_COL2IM)
        col2im_1ch_nchw_cython(
            dy_cols,
            dx,  # type: ignore
            dy.shape[0],
            self.hi,
            self.wi,
            self.ci,
            self.kh,
            self.kw,
            self.ho,
            self.wo,
            self.hpadding,
            self.wpadding,
            self.hstride,
            self.wstride,
            self.hdilation,
            self.wdilation,
        )
        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, PYDTNN_EVENT_FINISHED)
        dx: np.ndarray = dx.reshape((-1, self.ci, self.hi, self.wi))
        return dx
