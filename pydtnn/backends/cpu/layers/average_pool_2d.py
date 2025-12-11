import numpy as np
from pydtnn.backends.cpu.utils.average_pool_2d_nchw_cython import average_pool_2d_bwd_nchw_cython, average_pool_2d_fwd_nchw_cython
from pydtnn.backends.cpu.utils.average_pool_2d_nhwc_cython import average_pool_2d_bwd_nhwc_cython, average_pool_2d_fwd_nhwc_cython
from pydtnn.backends.cpu.utils.im2col_1ch_nchw_cython import col2im_1ch_nchw_cython, im2col_1ch_nchw_cython
from pydtnn.backends.cpu.utils.im2row_1ch_nhwc_cython import im2row_1ch_nhwc_cython, row2im_1ch_nhwc_cython
from pydtnn.backends.cpu.layers.abstract.pool_2d_layer import AbstractPool2DLayerCPU
from pydtnn.layers.average_pool_2d import AveragePool2D

from pydtnn.tracers.events import PYDTNN_OPS_EVENT, PYDTNN_OPS_EVENTS, PYDTNN_EVENT_FINISHED, PYDTNN_OPS_EVENT_enum


class AveragePool2DCPU(AveragePool2D[np.ndarray], AbstractPool2DLayerCPU):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.y: np.ndarray

    def initialize(self, prev_shape, x: np.ndarray | None = None):
        super().initialize(prev_shape, x)
        y_shape = self.model.encode_shape((self.model.batch_size, self.co, self.ho, self.wo))
        # NOTE: This attribute only stores data, its value before the operation doesn't matter; it's initalized due avoid warnings in "LayerAndActivationBase.export".
        self.y = np.zeros(y_shape, dtype=self.model.dtype, order="C")


    def _forward_nhwc_i2c(self, x: np.ndarray) -> np.ndarray:

        x_rows = np.zeros((x.shape[0] * self.ci * self.ho * self.wo, self.kh * self.kw), dtype=self.model.dtype, order="C")
        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, self.id * PYDTNN_OPS_EVENTS + PYDTNN_OPS_EVENT_enum.FORWARD_IM2COL)
        im2row_1ch_nhwc_cython(x, x_rows,
                               self.kh, self.kw, self.ho, self.wo,
                               self.vpadding, self.hpadding,
                               self.vstride, self.hstride, self.vdilation, self.hdilation)
        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, PYDTNN_EVENT_FINISHED)
        y: np.ndarray = np.mean(x_rows, axis=1, dtype=self.model.dtype)
        return y.reshape((-1, self.ho, self.wo, self.co), order="C", copy=None)

    def _forward_nhwc_cython(self, x: np.ndarray) -> np.ndarray:

        y = self.y[:x.shape[0], :]
        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, self.id * PYDTNN_OPS_EVENTS + PYDTNN_OPS_EVENT_enum.FORWARD_IM2COL)
        average_pool_2d_fwd_nhwc_cython(x, y,
                                        self.kh, self.kw, self.ho, self.wo,
                                        self.vpadding, self.hpadding,
                                        self.vstride, self.hstride,
                                        self.vdilation, self.hdilation)
        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, PYDTNN_EVENT_FINISHED)
        return np.asarray(y, dtype=self.model.dtype, order='C', copy=None)

    def _forward_nchw_i2c(self, x: np.ndarray) -> np.ndarray:
        n, c, _, _ = x.shape
        x_cols = np.zeros((self.kh * self.kw, n * c * self.ho * self.wo), dtype=self.model.dtype, order="C")

        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, self.id * PYDTNN_OPS_EVENTS + PYDTNN_OPS_EVENT_enum.FORWARD_IM2COL)
        im2col_1ch_nchw_cython(x, x_cols,
                               self.kh, self.kw, self.ho, self.wo,
                               self.vpadding, self.hpadding,
                               self.vstride, self.hstride, self.vdilation, self.hdilation)
        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, PYDTNN_EVENT_FINISHED)
        y: np.ndarray = np.mean(x_cols, axis=1, dtype=self.model.dtype)
        return y.reshape((-1, self.co, self.ho, self.wo), order="C", copy=None)

    def _forward_nchw_cython(self, x: np.ndarray) -> np.ndarray:
        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, self.id * PYDTNN_OPS_EVENTS + PYDTNN_OPS_EVENT_enum.FORWARD_IM2COL)
        y = self.y[:x.shape[0], :]
        average_pool_2d_fwd_nchw_cython(x, y,
                                        self.kh, self.kw, self.ho, self.wo,
                                        self.vpadding, self.hpadding,
                                        self.vstride, self.hstride,
                                        self.vdilation, self.hdilation)
        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, PYDTNN_EVENT_FINISHED)
        return np.asarray(y, dtype=self.model.dtype, order='C', copy=None)

    def _backward_nhwc_i2c(self, dy: np.ndarray) -> np.ndarray:
        pool_size = np.prod(self.pool_shape)
        dy_rows = np.tile(dy.reshape(-1, 1, copy=False) / pool_size, (1, pool_size))  # type: ignore (it is correct.)
        dx = np.zeros_like(dy, dtype=self.model.dtype)

        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, self.id * PYDTNN_OPS_EVENTS + PYDTNN_OPS_EVENT_enum.COMP_DX_COL2IM)
        row2im_1ch_nhwc_cython(dy_rows, dx,
                               dy.shape[0], self.hi, self.wi, self.ci,
                               self.kh, self.kw, self.ho, self.wo,
                               self.vpadding, self.hpadding,
                               self.vstride, self.hstride, self.vdilation, self.hdilation)
        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, PYDTNN_EVENT_FINISHED)
        return dx.reshape((-1, self.hi, self.wi, self.ci), order="C", copy=None)

    def _backward_nhwc_cython(self, dy: np.ndarray) -> np.ndarray:
        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, self.id * PYDTNN_OPS_EVENTS + PYDTNN_OPS_EVENT_enum.COMP_DX_COL2IM)
        # NOTE: It's necessary a new zero-initalized "dx" in every call since may be some values that are not re-set in the cython's function.
        dx = np.zeros((dy.shape[0], self.hi, self.wi, self.ci), dtype=self.model.dtype)
        average_pool_2d_bwd_nhwc_cython(dy, dx,
                                        dy.shape[0], self.hi, self.wi, self.ci,
                                        self.kh, self.kw, self.ho, self.wo,
                                        self.vpadding, self.hpadding,
                                        self.vstride, self.hstride,
                                        self.vdilation, self.hdilation)
        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, PYDTNN_EVENT_FINISHED)
        return np.asarray(dx, dtype=self.model.dtype, order='C', copy=None)

    def _backward_nchw_i2c(self, dy: np.ndarray) -> np.ndarray:
        pool_size = np.prod(self.pool_shape)
        dy_cols = np.tile(dy.flatten() / pool_size, (pool_size, 1))  # type: ignore (it is correct.)
        dy_cols = np.asarray(dy_cols, dtype=self.model.dtype, order="C", copy=None)
        dx = np.zeros((dy.shape[0], self.hi, self.wi, self.ci), dtype=self.model.dtype, order="C")

        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, self.id * PYDTNN_OPS_EVENTS + PYDTNN_OPS_EVENT_enum.COMP_DX_COL2IM)
        col2im_1ch_nchw_cython(dy_cols, dx,
                               dy.shape[0], self.hi, self.wi, self.ci,
                               self.kh, self.kw, self.ho, self.wo,
                               self.vpadding, self.hpadding,
                               self.vstride, self.hstride, 
                               self.vdilation, self.hdilation)
        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, PYDTNN_EVENT_FINISHED)
        return dx.reshape((-1, self.ci, self.hi, self.wi), order="C", copy=None)

    def _backward_nchw_cython(self, dy: np.ndarray) -> np.ndarray:
        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, self.id * PYDTNN_OPS_EVENTS + PYDTNN_OPS_EVENT_enum.COMP_DX_COL2IM)
        # NOTE: It's necessary a new zero-initalized "dx" in every call since may be some values that are not re-set in the cython's function.
        dx = np.zeros((dy.shape[0], self.ci, self.hi, self.wi), dtype=self.model.dtype, order="C")
        average_pool_2d_bwd_nchw_cython(dy, dx,
                                        dy.shape[0], self.hi, self.wi, self.ci,
                                        self.kh, self.kw, self.ho, self.wo,
                                        self.vpadding, self.hpadding,
                                        self.vstride, self.hstride,
                                        self.vdilation, self.hdilation)
        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, PYDTNN_EVENT_FINISHED)
        return np.asarray(dx, dtype=self.model.dtype, order='C', copy=None)
