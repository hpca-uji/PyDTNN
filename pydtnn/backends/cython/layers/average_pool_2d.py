"""Average pooling 2D layer implementation using Cython backends."""

import logging
from typing import TYPE_CHECKING

from pydtnn.backends.cython.layers.abstract.pool_2d_layer import AbstractPool2DLayerCython
from pydtnn.backends.cython.utils.average_pool_2d_nchw_cython import average_pool_2d_bwd_nchw_cython, average_pool_2d_fwd_nchw_cython
from pydtnn.backends.cython.utils.average_pool_2d_nhwc_cython import average_pool_2d_bwd_nhwc_cython, average_pool_2d_fwd_nhwc_cython
from pydtnn.backends.cython.utils.im2col_1ch_nchw_cython import col2im_1ch_nchw_cython, im2col_1ch_nchw_cython
from pydtnn.backends.cython.utils.im2row_1ch_nhwc_cython import im2row_1ch_nhwc_cython, row2im_1ch_nhwc_cython
from pydtnn.backends.numpy.layers.average_pool_2d import AveragePool2DNumpy
from pydtnn.libs import numpy as np
from pydtnn.tracers.events import PYDTNN_EVENT_FINISHED, PYDTNN_OPS_EVENT, PYDTNN_OPS_EVENTS, PYDTNN_OPS_EVENT_enum

__all__ = ("AveragePool2DCython",)


logger = logging.getLogger(__name__)


if TYPE_CHECKING:
    import numpy as np


class AveragePool2DCython(AveragePool2DNumpy, AbstractPool2DLayerCython):
    """Cython implementation of the 2D average pooling layer."""

    # CYTHON

    def _fwd_avg_pool_nchw(self, x: np.ndarray, y: np.ndarray) -> None:
        """Perform forward average pooling in NCHW format using Cython."""
        average_pool_2d_fwd_nchw_cython(
            x,
            y,  # type: ignore
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

    def _fwd_avg_pool_nhwc(self, x: np.ndarray, y: np.ndarray) -> None:
        """Perform forward average pooling in NHWC format using Cython."""
        average_pool_2d_fwd_nhwc_cython(
            x,
            y,  # type: ignore
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

    def _bwd_avg_pool_nhwc(self, dx: np.ndarray, dy: np.ndarray) -> None:
        """Perform backward average pooling in NHWC format using Cython."""
        average_pool_2d_bwd_nhwc_cython(
            dy,
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

    def _bwd_avg_pool_nchw(self, dx: np.ndarray, dy: np.ndarray) -> None:
        """Perform backward average pooling in NCHW format using Cython."""
        average_pool_2d_bwd_nchw_cython(
            dy,
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
        """Perform forward pass in NHWC format using im2row transformation."""
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
        y: np.ndarray = np.mean(x_rows, axis=1, dtype=self.model.dtype)
        return y.reshape((-1, self.ho, self.wo, self.co))

    def _forward_nchw_i2c(self, x: np.ndarray) -> np.ndarray:
        """Perform forward pass in NCHW format using im2col transformation."""
        n, c, _, _ = x.shape
        x_cols: np.ndarray = np.zeros((self.kh * self.kw, n * c * self.ho * self.wo), dtype=self.model.dtype)

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
        y: np.ndarray = np.mean(x_cols, axis=1, dtype=self.model.dtype)
        return y.reshape((-1, self.co, self.ho, self.wo))

    def _backward_nhwc_i2c(self, dy: np.ndarray) -> np.ndarray:
        """Perform backward pass in NHWC format using row2im transformation."""
        pool_size = np.prod(self.pool_shape)
        dy_rows: np.ndarray = np.tile(dy.reshape(-1, 1, copy=False) / pool_size, (1, pool_size))  # type: ignore (it is correct.)
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
        """Perform backward pass in NCHW format using col2im transformation."""
        pool_size = np.prod(self.pool_shape)
        dy_cols: np.ndarray = np.tile(dy.flatten() / pool_size, (pool_size, 1))  # type: ignore (it is correct.)
        dy_cols: np.ndarray = np.asarray(dy_cols, dtype=self.model.dtype, order="C")
        dx: np.ndarray = np.zeros((dy.shape[0], self.hi, self.wi, self.ci), dtype=self.model.dtype)

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
        return dx.reshape((-1, self.ci, self.hi, self.wi))
