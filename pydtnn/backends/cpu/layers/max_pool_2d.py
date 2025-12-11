import numpy as np
from pydtnn.backends.cpu.utils.argmax_cython import argmax_cython
from pydtnn.backends.cpu.utils.im2col_1ch_nchw_cython import col2im_1ch_nchw_cython, im2col_1ch_nchw_cython
from pydtnn.backends.cpu.utils.im2row_1ch_nhwc_cython import im2row_1ch_nhwc_cython, row2im_1ch_nhwc_cython
from pydtnn.backends.cpu.utils.max_pool_2d_nchw_cython import max_pool_2d_bwd_nchw_cython, max_pool_2d_fwd_nchw_cython
from pydtnn.backends.cpu.utils.max_pool_2d_nhwc_cython import max_pool_2d_bwd_nhwc_cython, max_pool_2d_fwd_nhwc_cython

from pydtnn.backends.cpu.layers.abstract.pool_2d_layer import AbstractPool2DLayerCPU
from pydtnn.layers.max_pool_2d import MaxPool2D
from pydtnn.model import Model
from pydtnn.tracers.events import PYDTNN_OPS_EVENT, PYDTNN_OPS_EVENTS, PYDTNN_EVENT_FINISHED, PYDTNN_OPS_EVENT_enum
from pydtnn.utils.constants import ArrayShape


class MaxPool2DCPU(MaxPool2D[np.ndarray], AbstractPool2DLayerCPU):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # The following attribute will be intialized later.
        self.idx_max: np.ndarray = None # type: ignore
        self.y: np.ndarray

    def initialize(self, prev_shape: ArrayShape, x: np.ndarray | None = None):
        super().initialize(prev_shape, x)
        self.minval = np.iinfo(self.model.dtype).min if np.issubdtype(self.model.dtype, np.integer) else np.finfo(self.model.dtype).min
        idx_max_shape = self.model.encode_shape((self.model.batch_size, self.co, self.ho, self.wo))
        # NOTE: This attribute only stores data, its value before the operation doesn't matter; it's initalized due avoid warnings in "LayerAndActivationBase.export".
        self._idx_max = np.zeros(idx_max_shape, dtype=np.int32)

    def _forward_nhwc_i2c(self, x: np.ndarray) -> np.ndarray:
        y = np.zeros((x.shape[0],), dtype=self.model.dtype, order="C")
        amax = np.zeros((x.shape[0],), dtype=np.int32, order="C")
        rng = np.zeros((x.shape[0],), dtype=np.int32, order="C")
        x_rows = np.zeros((x.shape[0] * self.ci * self.ho * self.wo, self.kh * self.kw), dtype=self.model.dtype, order="C")

        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, self.id * PYDTNN_OPS_EVENTS + PYDTNN_OPS_EVENT_enum.FORWARD_IM2COL)
        im2row_1ch_nhwc_cython(x, x_rows,
                               self.kh, self.kw, self.ho, self.wo,
                               self.vpadding, self.hpadding,
                               self.vstride, self.hstride, self.vdilation, self.hdilation)
        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, PYDTNN_EVENT_FINISHED)
        idx_max = argmax_cython(x_rows, y, amax, rng, axis=1)

        idx_max: np.ndarray
        if self.model.mode is Model.Mode.TRAIN:
            self.idx_max = idx_max
        return y.reshape((-1, self.ho, self.wo, self.co), order="C", copy=None)

    def _forward_nhwc_cython(self, x: np.ndarray) -> np.ndarray:

        y = self.y[:x.shape[0], :]
        self.idx_max = self._idx_max[:x.shape[0], :]
        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, self.id * PYDTNN_OPS_EVENTS + PYDTNN_OPS_EVENT_enum.FORWARD_IM2COL)
        max_pool_2d_fwd_nhwc_cython(x, y, self.idx_max,
                                    self.kh, self.kw, self.ho, self.wo,
                                    self.vpadding, self.hpadding,
                                    self.vstride, self.hstride,
                                    self.vdilation, self.hdilation,
                                    self.minval)
        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, PYDTNN_EVENT_FINISHED)
        return y

    def _forward_nchw_i2c(self, x: np.ndarray) -> np.ndarray:
        n, c, _, _ = x.shape
        x_cols = np.zeros((self.kh * self.kw, n * c * self.ho * self.wo), dtype=self.model.dtype, order="C")
        y = np.zeros((n,), dtype=self.model.dtype, order="C")
        amax = np.zeros((n,), dtype=np.int32, order="C")
        rng = np.zeros((n,), dtype=np.int32, order="C")

        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, self.id * PYDTNN_OPS_EVENTS + PYDTNN_OPS_EVENT_enum.FORWARD_IM2COL)
        im2col_1ch_nchw_cython(x, x_cols,
                               self.kh, self.kw, self.ho, self.wo,
                               self.vpadding, self.hpadding,
                               self.vstride, self.hstride, self.vdilation, self.hdilation)
        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, PYDTNN_EVENT_FINISHED)
        idx_max = argmax_cython(x_cols, y, amax, rng, axis=0)
        idx_max: np.ndarray
        if self.model.mode is Model.Mode.TRAIN:
            self.idx_max = idx_max
        return y.reshape((-1, self.co, self.ho, self.wo), order="C", copy=None)

    def _forward_nchw_cython(self, x: np.ndarray) -> np.ndarray:
        y = self.y[:x.shape[0], :]
        self.idx_max = self._idx_max[:x.shape[0], :]

        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, self.id * PYDTNN_OPS_EVENTS + PYDTNN_OPS_EVENT_enum.FORWARD_IM2COL)
        max_pool_2d_fwd_nchw_cython(x, y, self.idx_max,
                                    self.kh, self.kw, self.ho, self.wo,
                                    self.vpadding, self.hpadding,
                                    self.vstride, self.hstride,
                                    self.vdilation, self.hdilation,
                                    self.minval)
        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, PYDTNN_EVENT_FINISHED)
        return np.asarray(y, dtype=self.model.dtype, order='C', copy=None)

    def _backward_nhwc_i2c(self, dy: np.ndarray) -> np.ndarray:
        dy_rows = np.zeros((np.prod(dy.shape), self.kh * self.kw), dtype=self.model.dtype, order="C")
        dy_rows[self.idx_max] = dy.flatten()
        dx = np.zeros_like(dy, dtype=self.model.dtype, order="C")
        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, self.id * PYDTNN_OPS_EVENTS + PYDTNN_OPS_EVENT_enum.COMP_DX_COL2IM)
        row2im_1ch_nhwc_cython(dy_rows, dx,
                               dy.shape[0], self.hi, self.wi, self.ci,
                               self.kh, self.kw, self.ho, self.wo,
                               self.vpadding, self.hpadding,
                               self.vstride, self.hstride, self.vdilation, self.hdilation)
        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, PYDTNN_EVENT_FINISHED)
        return dx.reshape((-1, self.hi, self.wi, self.ci), order="C", copy=None)

    def _backward_nhwc_cython(self, dy: np.ndarray) -> np.ndarray:
        dx = np.zeros((dy.shape[0], self.hi, self.wi, self.ci), dtype=self.model.dtype, order="C")
        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, self.id * PYDTNN_OPS_EVENTS + PYDTNN_OPS_EVENT_enum.COMP_DX_COL2IM)
        max_pool_2d_bwd_nhwc_cython(dy, self.idx_max, dx,
                                    dy.shape[0], self.hi, self.wi, self.ci,
                                    self.kh, self.kw, self.ho, self.wo,
                                    self.vpadding, self.hpadding,
                                    self.vstride, self.hstride, self.vdilation, self.hdilation)
        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, PYDTNN_EVENT_FINISHED)
        return dx

    def _backward_nchw_i2c(self, dy: np.ndarray) -> np.ndarray:
        dy_cols = np.zeros((self.kh * self.kw, np.prod(dy.shape)), dtype=self.model.dtype, order="C")
        dy_cols[self.idx_max] = dy.flatten(order="C").view(dtype=self.model.dtype)
        dx = np.zeros((dy.shape[0], self.ci, self.hi, self.wi), dtype=self.model.dtype, order="C")

        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, self.id * PYDTNN_OPS_EVENTS + PYDTNN_OPS_EVENT_enum.COMP_DX_COL2IM)
        col2im_1ch_nchw_cython(dy_cols, dx,
                               dy.shape[0], self.hi, self.wi, self.ci,
                               self.kh, self.kw, self.ho, self.wo,
                               self.vpadding, self.hpadding,
                               self.vstride, self.hstride, self.vdilation, self.hdilation)
        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, PYDTNN_EVENT_FINISHED)
        dx: np.ndarray = dx.reshape((-1, self.ci, self.hi, self.wi), order="C", copy=None)
        return dx

    def _backward_nchw_cython(self, dy: np.ndarray) -> np.ndarray:

        dx = np.zeros((dy.shape[0], self.ci, self.hi, self.wi), dtype=self.model.dtype, order="C")
        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, self.id * PYDTNN_OPS_EVENTS + PYDTNN_OPS_EVENT_enum.COMP_DX_COL2IM)
        max_pool_2d_bwd_nchw_cython(dy, self.idx_max, dx,
                                    dy.shape[0], self.hi, self.wi, self.ci,
                                    self.kh, self.kw, self.ho, self.wo,
                                    self.vpadding, self.hpadding,
                                    self.vstride, self.hstride, self.vdilation, self.hdilation)
        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, PYDTNN_EVENT_FINISHED)
        return np.asarray(dx, dtype=self.model.dtype, order='C', copy=None)
