#
#  This file is part of Python Distributed Training of Neural Networks (PyDTNN)
#
#  Copyright (C) 2021-25 Universitat Jaume I
#
#  PyDTNN is free software: you can redistribute it and/or modify it under the
#  terms of the GNU General Public License as published by the Free Software
#  Foundation, either version 3 of the License, or (at your option) any later
#  version.
#
#  This program is distributed in the hope that it will be useful, but WITHOUT
#  ANY WARRANTY; without even the implied warranty of MERCHANTABILITY
#  or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public
#  License for more details.
#
#  You should have received a copy of the GNU General Public License along
#  with this program. If not, see <https://www.gnu.org/licenses/>.
#

import numpy as np
from pydtnn.backends.cpu.layers.abstract_pool_2d_layer_cpu import AbstractPool2DLayerCPU
from pydtnn.layers import AveragePool2D
from pydtnn.utils import PYDTNN_TENSOR_FORMAT

from pydtnn.cython_modules import im2row_1ch_nhwc_cython, row2im_1ch_nhwc_cython, \
                                  im2col_1ch_nchw_cython, col2im_1ch_nchw_cython, \
                                  average_pool_2d_fwd_nhwc_cython, average_pool_2d_bwd_nhwc_cython, \
                                  average_pool_2d_fwd_nchw_cython, average_pool_2d_bwd_nchw_cython
from pydtnn.tracers import PYDTNN_OPS_EVENT, PYDTNN_OPS_EVENTS, PYDTNN_EVENT_FINISHED, PYDTNN_OPS_EVENT_enum, PYDTNN_OPS_EVENT_enum

class AveragePool2DCPU(AbstractPool2DLayerCPU, AveragePool2D):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def initialize(self, prev_shape):
        super().initialize(prev_shape)
        match self.model.tensor_format:
            case PYDTNN_TENSOR_FORMAT.NHWC:
                self.y = np.empty((self.model.batch_size, self.ho, self.wo, self.co), dtype=self.model.dtype)
            case PYDTNN_TENSOR_FORMAT.NCHW:
                self.y = np.empty((self.model.batch_size, self.co, self.ho, self.wo), dtype=self.model.dtype)
            case _:
                raise NotImplementedError(f"\"AveragePool2DCPU\" layer is not implemted for the format: {self.model.tensor_format}")


    def _forward_nhwc_i2c(self, x:np.ndarray) -> np.ndarray:

        x_rows = np.zeros((x.shape[0] * self.ci * self.ho * self.wo, self.kh * self.kw), dtype=self.model.dtype)
        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, self.id * PYDTNN_OPS_EVENTS + PYDTNN_OPS_EVENT_enum.FORWARD_IM2COL)
        im2row_1ch_nhwc_cython(x, x_rows,
                               self.kh, self.kw, self.ho, self.wo,
                               self.vpadding, self.hpadding,
                               self.vstride, self.hstride, self.vdilation, self.hdilation)
        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, PYDTNN_EVENT_FINISHED)
        y:np.ndarray = np.mean(x_rows, axis=1)
        return y.reshape((-1, self.ho, self.wo, self.co), order="C", copy=None)

    def _forward_nhwc_cython(self, x:np.ndarray) -> np.ndarray:

        y = self.y[:x.shape[0], :]        
        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, self.id * PYDTNN_OPS_EVENTS + PYDTNN_OPS_EVENT_enum.FORWARD_IM2COL)        
        average_pool_2d_fwd_nhwc_cython(x, y, 
                                        self.kh, self.kw, self.ho, self.wo, 
                                        self.vpadding, self.hpadding,
                                        self.vstride, self.hstride, 
                                        self.vdilation, self.hdilation)
        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, PYDTNN_EVENT_FINISHED)
        return y

    def _forward_nchw_i2c(self, x:np.ndarray) -> np.ndarray:
        n, c, _, _ = x.shape
        x_cols = np.zeros((self.kh * self.kw, n * c * self.ho * self.wo), dtype=self.model.dtype)

        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, self.id * PYDTNN_OPS_EVENTS + PYDTNN_OPS_EVENT_enum.FORWARD_IM2COL)
        im2col_1ch_nchw_cython(x, x_cols,
                               self.kh, self.kw, self.ho, self.wo,
                               self.vpadding, self.hpadding,
                               self.vstride, self.hstride, self.vdilation, self.hdilation)
        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, PYDTNN_EVENT_FINISHED)
        y:np.ndarray = np.mean(x_cols, axis=0)
        return y.reshape((-1, self.co, self.ho, self.wo), order="C", copy=True)

    def _forward_nchw_cython(self, x:np.ndarray) -> np.ndarray:
        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, self.id * PYDTNN_OPS_EVENTS + PYDTNN_OPS_EVENT_enum.FORWARD_IM2COL)
        y = self.y[:x.shape[0], :]
        average_pool_2d_fwd_nchw_cython(x, y,
                                        self.kh, self.kw, self.ho, self.wo, 
                                        self.vpadding, self.hpadding,
                                        self.vstride, self.hstride, 
                                        self.vdilation, self.hdilation)
        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, PYDTNN_EVENT_FINISHED)
        return y

    def _backward_nhwc_i2c(self, dy:np.ndarray) -> np.ndarray:
        pool_size = np.prod(self.pool_shape)
        dy_rows = np.tile(dy.reshape(-1, 1, copy=False) / pool_size, (1, pool_size))
        dx = np.zeros_like(dy, dtype=self.model.dtype)

        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, self.id * PYDTNN_OPS_EVENTS + PYDTNN_OPS_EVENT_enum.COMP_DX_COL2IM)
        row2im_1ch_nhwc_cython(dy_rows, dx,
                                dy.shape[0], self.hi, self.wi, self.ci,
                                self.kh, self.kw, self.ho, self.wo,
                                self.vpadding, self.hpadding,
                                self.vstride, self.hstride, self.vdilation, self.hdilation)
        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, PYDTNN_EVENT_FINISHED)
        return dx.reshape((-1, self.hi, self.wi, self.ci), order="C", copy=True)

    def _backward_nhwc_cython(self, dy:np.ndarray) -> np.ndarray:
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
        return dx

    def _backward_nchw_i2c(self, dy:np.ndarray) -> np.ndarray:
        pool_size = np.prod(self.pool_shape)
        dy_cols = np.tile(dy.flatten() / pool_size, (pool_size, 1))
        dx = np.zeros((dy.shape[0], self.hi, self.wi, self.ci), dtype= self.model.dtype)

        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, self.id * PYDTNN_OPS_EVENTS + PYDTNN_OPS_EVENT_enum.COMP_DX_COL2IM)
        col2im_1ch_nchw_cython(dy_cols, dx,
                                dy.shape[0], self.hi, self.wi, self.ci,
                                self.kh, self.kw, self.ho, self.wo, 
                                self.vpadding, self.hpadding,
                                self.vstride, self.hstride, self.vdilation, self.hdilation)
        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, PYDTNN_EVENT_FINISHED)
        dx = dx.reshape((-1, self.ci, self.hi, self.wi), copy=False)
        return dx

    def _backward_nchw_cython(self, dy:np.ndarray) -> np.ndarray:
        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, self.id * PYDTNN_OPS_EVENTS + PYDTNN_OPS_EVENT_enum.COMP_DX_COL2IM)
        # NOTE: It's necessary a new zero-initalized "dx" in every call since may be some values that are not re-set in the cython's function.
        dx = np.zeros((dy.shape[0], self.ci, self.hi, self.wi), dtype=self.model.dtype)
        average_pool_2d_bwd_nchw_cython(dy, dx, 
                                        dy.shape[0], self.hi, self.wi, self.ci,
                                        self.kh, self.kw, self.ho, self.wo,
                                        self.vpadding, self.hpadding,
                                        self.vstride, self.hstride, 
                                        self.vdilation, self.hdilation)
        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, PYDTNN_EVENT_FINISHED)
        return dx
