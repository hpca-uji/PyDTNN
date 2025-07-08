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
from pydtnn.cython_modules import im2row_1ch_nhwc_cython, row2im_1ch_nhwc_cython, \
                                  im2col_1ch_nchw_cython, col2im_1ch_nchw_cython, argmax_cython, \
                                  max_pool_2d_fwd_nhwc_cython, max_pool_2d_bwd_nhwc_cython, \
                                  max_pool_2d_fwd_nchw_cython, max_pool_2d_bwd_nchw_cython
from pydtnn.layers import MaxPool2D
from pydtnn.model import TRAIN_MODE
from pydtnn.tracers import PYDTNN_OPS_EVENT, PYDTNN_OPS_EVENTS, PYDTNN_EVENT_FINISHED, PYDTNN_OPS_EVENT_enum
from pydtnn.utils import PYDTNN_TENSOR_FORMAT

class MaxPool2DCPU(AbstractPool2DLayerCPU, MaxPool2D):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.idx_max:np.ndarray = None

    def initialize(self, prev_shape):
        super().initialize(prev_shape)
        self.minval = np.iinfo(self.model.dtype).min if np.issubdtype(self.model.dtype, np.integer) else np.finfo(self.model.dtype).min            
        
        match self.model.tensor_format:
            case PYDTNN_TENSOR_FORMAT.NCHW:
                self._idx_max = np.empty((self.model.batch_size, self.co, self.ho, self.wo), dtype=np.int32)
            case PYDTNN_TENSOR_FORMAT.NHWC:
                self._idx_max = np.empty((self.model.batch_size, self.ho, self.wo, self.co), dtype=np.int32)
            case _:
                raise TypeError(f"Function: \'AveragePool2DCPU\'. Error:\n\tFormat: \'{self.model.tensor_format}\' not supported.")

    def _forward_nhwc_i2c(self, x: np.ndarray) -> np.ndarray:
        y = np.zeros((x.shape[0],), dtype=self.model.dtype)
        amax = np.zeros((x.shape[0],), dtype=np.int32)
        rng = np.zeros((x.shape[0],), dtype=np.int32)
        x_rows = np.zeros((x.shape[0] * self.ci * self.ho * self.wo, self.kh * self.kw), dtype=self.model.dtype)

        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, self.id * PYDTNN_OPS_EVENTS + PYDTNN_OPS_EVENT_enum.FORWARD_IM2COL)        
        im2row_1ch_nhwc_cython(x, x_rows,
                               self.kh, self.kw, self.ho, self.wo,
                               self.vpadding, self.hpadding,
                               self.vstride, self.hstride, self.vdilation, self.hdilation)
        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, PYDTNN_EVENT_FINISHED)
        idx_max = argmax_cython(x_rows, y, amax, rng, axis=1)        
        
        idx_max:np.ndarray
        if self.model.mode == TRAIN_MODE:
            self.idx_max = idx_max
        return y.reshape((-1, self.ho, self.wo, self.co), copy=False)

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
        x_cols = np.zeros((self.kh * self.kw, n * c * self.ho * self.wo), dtype=self.model.dtype)
        y = np.zeros((n,), dtype=self.model.dtype)
        amax = np.zeros((n,), dtype=np.int32)
        rng = np.zeros((n,), dtype=np.int32)

        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, self.id * PYDTNN_OPS_EVENTS + PYDTNN_OPS_EVENT_enum.FORWARD_IM2COL)
        im2col_1ch_nchw_cython(x, x_cols,
                               self.kh, self.kw, self.ho, self.wo,
                               self.vpadding, self.hpadding,
                               self.vstride, self.hstride, self.vdilation, self.hdilation)
        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, PYDTNN_EVENT_FINISHED)
        idx_max = argmax_cython(x_cols, y, amax, rng, axis=0)
        idx_max: np.ndarray
        if self.model.mode == TRAIN_MODE:
            self.idx_max = idx_max
        return y.reshape((-1, self.co, self.ho, self.wo), copy=False)

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
        return y

    def _backward_nhwc_i2c(self, dy: np.ndarray) -> np.ndarray:
        dy_rows = np.zeros((np.prod(dy.shape), self.kh * self.kw), dtype=self.model.dtype)
        dy_rows[self.idx_max] = dy.flatten()
        dx = np.zeros_like(dy, dtype=self.model.dtype)
        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, self.id * PYDTNN_OPS_EVENTS + PYDTNN_OPS_EVENT_enum.COMP_DX_COL2IM)
        row2im_1ch_nhwc_cython(dy_rows, dx,
                                dy.shape[0], self.hi, self.wi, self.ci,
                                self.kh, self.kw, self.ho, self.wo,
                                self.vpadding, self.hpadding,
                                self.vstride, self.hstride, self.vdilation, self.hdilation)
        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, PYDTNN_EVENT_FINISHED)
        return dx.reshape((-1, self.hi, self.wi, self.ci), copy=False)

    def _backward_nhwc_cython(self, dy: np.ndarray) -> np.ndarray:
        # NOTE: It's necessary to initalize dx with "zeros" in every call due there are some position that "max_pool_2d_bwd_nhwc_cython" doesn't set.
        dx = np.zeros((dy.shape[0], self.hi, self.wi, self.ci), dtype= self.model.dtype)
        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, self.id * PYDTNN_OPS_EVENTS + PYDTNN_OPS_EVENT_enum.COMP_DX_COL2IM)
        max_pool_2d_bwd_nhwc_cython(dy, self.idx_max, dx,
                                    dy.shape[0], self.hi, self.wi, self.ci,
                                    self.kh, self.kw, self.ho, self.wo,
                                    self.vpadding, self.hpadding,
                                    self.vstride, self.hstride, self.vdilation, self.hdilation)
        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, PYDTNN_EVENT_FINISHED)
        return dx

    def _backward_nchw_i2c(self, dy: np.ndarray) -> np.ndarray:
        dy_cols = np.zeros((self.kh * self.kw, np.prod(dy.shape)), dtype=self.model.dtype)
        dy_cols[self.idx_max] = dy.flatten().astype(dtype=self.model.dtype, copy=False)
        dx = np.zeros((dy.shape[0], self.ci, self.hi, self.wi), dtype= self.model.dtype)

        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, self.id * PYDTNN_OPS_EVENTS + PYDTNN_OPS_EVENT_enum.COMP_DX_COL2IM)
        col2im_1ch_nchw_cython(dy_cols, dx,
                                dy.shape[0], self.hi, self.wi, self.ci,
                                self.kh, self.kw, self.ho, self.wo, 
                                self.vpadding, self.hpadding,
                                self.vstride, self.hstride, self.vdilation, self.hdilation)
        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, PYDTNN_EVENT_FINISHED)
        dx:np.ndarray = dx.reshape((-1, self.ci, self.hi, self.wi), copy=False)
        return dx

    def _backward_nchw_cython(self, dy: np.ndarray) -> np.ndarray:
        # NOTE: It's necessary to initalize dx with "zeros" in every call due there are some position that "max_pool_2d_bwd_nhwc_cython" doesn't set.
        dx = np.zeros((dy.shape[0], self.ci, self.hi, self.wi), dtype= self.model.dtype)
        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, self.id * PYDTNN_OPS_EVENTS + PYDTNN_OPS_EVENT_enum.COMP_DX_COL2IM)          
        max_pool_2d_bwd_nchw_cython(dy, self.idx_max, dx, 
                                    dy.shape[0], self.hi, self.wi, self.ci,
                                    self.kh, self.kw, self.ho, self.wo,
                                    self.vpadding, self.hpadding,
                                    self.vstride, self.hstride, self.vdilation, self.hdilation)
        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, PYDTNN_EVENT_FINISHED)
        return dx

