from pydtnn.backends.numpy.layers.max_pool_2d import MaxPool2DNumpy
from pydtnn.libs import numpy as np
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    import numpy as np
from pydtnn.backends.cython.utils.argmax_cython import argmax_cython
from pydtnn.backends.cython.utils.im2col_1ch_nchw_cython import col2im_1ch_nchw_cython, im2col_1ch_nchw_cython
from pydtnn.backends.cython.utils.im2row_1ch_nhwc_cython import im2row_1ch_nhwc_cython, row2im_1ch_nhwc_cython
from pydtnn.backends.cython.utils.max_pool_2d_nchw_cython import max_pool_2d_bwd_nchw_cython, max_pool_2d_fwd_nchw_cython
from pydtnn.backends.cython.utils.max_pool_2d_nhwc_cython import max_pool_2d_bwd_nhwc_cython, max_pool_2d_fwd_nhwc_cython

from pydtnn.model import Model
from pydtnn.tracers.events import PYDTNN_OPS_EVENT, PYDTNN_OPS_EVENTS, PYDTNN_EVENT_FINISHED, PYDTNN_OPS_EVENT_enum


class MaxPool2DCython(MaxPool2DNumpy):
    ##############
    ### CYTHON ###
    ##############
    def _forward_nhwc(self, x: np.ndarray) -> np.ndarray:

        # y:np.ndarray = self.y[:x.shape[0], :]
        y = self.get_y(x.shape[0])
        self.idx_max: np.ndarray = self._idx_max[:x.shape[0], :]

        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, self.id * PYDTNN_OPS_EVENTS + PYDTNN_OPS_EVENT_enum.FORWARD_IM2COL)
        max_pool_2d_fwd_nhwc_cython(x, y, self.idx_max,
                                    self.kh, self.kw, self.ho, self.wo,
                                    self.hpadding, self.wpadding,
                                    self.hstride, self.wstride,
                                    self.hdilation, self.wdilation,
                                    self.minval)
        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, PYDTNN_EVENT_FINISHED)
        return y

    def _forward_nchw(self, x: np.ndarray) -> np.ndarray:
        # y:np.ndarray = self.y[:x.shape[0], :]
        y = self.get_y(x.shape[0])
        self.idx_max = self._idx_max[:x.shape[0], :]

        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, self.id * PYDTNN_OPS_EVENTS + PYDTNN_OPS_EVENT_enum.FORWARD_IM2COL)
        max_pool_2d_fwd_nchw_cython(x, y, self.idx_max,
                                    self.kh, self.kw, self.ho, self.wo,
                                    self.hpadding, self.wpadding,
                                    self.hstride, self.wstride,
                                    self.hdilation, self.wdilation,
                                    self.minval)
        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, PYDTNN_EVENT_FINISHED)
        return np.asarray(y, dtype=self.model.dtype, order="C")

    def _backward_nhwc(self, dy: np.ndarray) -> np.ndarray:
        # dx:np.ndarray = self.dx[ :dy.shape[0], :]
        dx = self.get_dx(dy.shape[0])
        dx.fill(0)
        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, self.id * PYDTNN_OPS_EVENTS + PYDTNN_OPS_EVENT_enum.COMP_DX_COL2IM)
        max_pool_2d_bwd_nhwc_cython(dy, self.idx_max, dx,
                                    dy.shape[0], self.hi, self.wi, self.ci,
                                    self.kh, self.kw, self.ho, self.wo,
                                    self.hpadding, self.wpadding,
                                    self.hstride, self.wstride, self.hdilation, self.wdilation)
        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, PYDTNN_EVENT_FINISHED)
        return dx

    def _backward_nchw(self, dy: np.ndarray) -> np.ndarray:

        # dx:np.ndarray = self.dx[ :dy.shape[0], :]
        dx = self.get_dx(dy.shape[0])
        dx.fill(0)
        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, self.id * PYDTNN_OPS_EVENTS + PYDTNN_OPS_EVENT_enum.COMP_DX_COL2IM)
        max_pool_2d_bwd_nchw_cython(dy, self.idx_max, dx,
                                    dy.shape[0], self.hi, self.wi, self.ci,
                                    self.kh, self.kw, self.ho, self.wo,
                                    self.hpadding, self.wpadding,
                                    self.hstride, self.wstride, self.hdilation, self.wdilation)
        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, PYDTNN_EVENT_FINISHED)
        return np.asarray(dx, dtype=self.model.dtype, order="C")

    ###########
    ### I2C ###
    ###########

    def _forward_nhwc_i2c(self, x: np.ndarray) -> np.ndarray:
        y: np.ndarray = np.zeros((x.shape[0],), dtype=self.model.dtype)
        amax: np.ndarray = np.zeros((x.shape[0],), dtype=np.int32)
        rng: np.ndarray = np.zeros((x.shape[0],), dtype=np.int32)
        x_rows: np.ndarray = np.zeros((x.shape[0] * self.ci * self.ho * self.wo, self.kh * self.kw), dtype=self.model.dtype)

        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, self.id * PYDTNN_OPS_EVENTS + PYDTNN_OPS_EVENT_enum.FORWARD_IM2COL)
        im2row_1ch_nhwc_cython(x, x_rows,
                               self.kh, self.kw, self.ho, self.wo,
                               self.hpadding, self.wpadding,
                               self.hstride, self.wstride, self.hdilation, self.wdilation)
        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, PYDTNN_EVENT_FINISHED)
        idx_max: np.ndarray = argmax_cython(x_rows, y, amax, rng, axis=1)

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
        im2col_1ch_nchw_cython(x, x_cols,
                               self.kh, self.kw, self.ho, self.wo,
                               self.hpadding, self.wpadding,
                               self.hstride, self.wstride, self.hdilation, self.wdilation)
        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, PYDTNN_EVENT_FINISHED)
        idx_max: np.ndarray = argmax_cython(x_cols, y, amax, rng, axis=0)
        if self.model.mode is Model.Mode.TRAIN:
            self.idx_max = idx_max
        return y.reshape((-1, self.co, self.ho, self.wo))

    def _backward_nhwc_i2c(self, dy: np.ndarray) -> np.ndarray:
        dy_rows: np.ndarray = np.zeros((np.prod(dy.shape), self.kh * self.kw), dtype=self.model.dtype)
        dy_rows[self.idx_max] = dy.flatten()
        dx: np.ndarray = np.zeros_like(dy, dtype=self.model.dtype)
        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, self.id * PYDTNN_OPS_EVENTS + PYDTNN_OPS_EVENT_enum.COMP_DX_COL2IM)
        row2im_1ch_nhwc_cython(dy_rows, dx,
                               dy.shape[0], self.hi, self.wi, self.ci,
                               self.kh, self.kw, self.ho, self.wo,
                               self.hpadding, self.wpadding,
                               self.hstride, self.wstride, self.hdilation, self.wdilation)
        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, PYDTNN_EVENT_FINISHED)
        return dx.reshape((-1, self.hi, self.wi, self.ci))

    def _backward_nchw_i2c(self, dy: np.ndarray) -> np.ndarray:
        dy_cols: np.ndarray = np.zeros((self.kh * self.kw, np.prod(dy.shape)), dtype=self.model.dtype)
        dy_cols[self.idx_max] = dy.flatten(order="C").view(dtype=self.model.dtype)
        dx: np.ndarray = np.zeros((dy.shape[0], self.ci, self.hi, self.wi), dtype=self.model.dtype)

        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, self.id * PYDTNN_OPS_EVENTS + PYDTNN_OPS_EVENT_enum.COMP_DX_COL2IM)
        col2im_1ch_nchw_cython(dy_cols, dx,
                               dy.shape[0], self.hi, self.wi, self.ci,
                               self.kh, self.kw, self.ho, self.wo,
                               self.hpadding, self.wpadding,
                               self.hstride, self.wstride, self.hdilation, self.wdilation)
        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, PYDTNN_EVENT_FINISHED)
        dx: np.ndarray = dx.reshape((-1, self.ci, self.hi, self.wi))
        return dx
