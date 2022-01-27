#
#  This file is part of Python Distributed Training of Neural Networks (PyDTNN)
#
#  Copyright (C) 2021-22 Universitat Jaume I
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
from pydtnn.tracers import PYDTNN_OPS_EVENT, PYDTNN_OPS_EVENTS, PYDTNN_OPS_COMP_DX_COL2IM, PYDTNN_OPS_FORWARD_IM2COL


class MaxPool2DCPU(AbstractPool2DLayerCPU, MaxPool2D):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.idx_max = None

    def _forward_nhwc_i2c(self, x):
        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, self.id * PYDTNN_OPS_EVENTS + PYDTNN_OPS_FORWARD_IM2COL)
        x_rows = im2row_1ch_nhwc_cython(x, self.kh, self.kw, self.vpadding, self.hpadding,
                                        self.vstride, self.hstride, self.vdilation, self.hdilation)
        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, 0)
        y, idx_max = argmax_cython(x_rows, axis=1)
        if self.model.mode == TRAIN_MODE:
            self.idx_max = idx_max
        return y.reshape(-1, self.ho, self.wo, self.co)

    def _forward_nhwc_cython(self, x):
        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, self.id * PYDTNN_OPS_EVENTS + PYDTNN_OPS_FORWARD_IM2COL)
        y, idx_max = max_pool_2d_fwd_nhwc_cython(x, self.kh, self.kw, self.vpadding, self.hpadding,
                                                 self.vstride, self.hstride, self.vdilation, self.hdilation)
        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, 0)
        if self.model.mode == TRAIN_MODE:
            self.idx_max = idx_max
        return y

    def _forward_nchw_i2c(self, x):
        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, self.id * PYDTNN_OPS_EVENTS + PYDTNN_OPS_FORWARD_IM2COL)
        x_cols = im2col_1ch_nchw_cython(x, self.kh, self.kw, self.vpadding, self.hpadding,
                                        self.vstride, self.hstride, self.vdilation, self.hdilation)
        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, 0)
        y, idx_max = argmax_cython(x_cols, axis=0)
        if self.model.mode == TRAIN_MODE:
            self.idx_max = idx_max
        return y.reshape(-1, self.co, self.ho, self.wo)

    def _forward_nchw_cython(self, x):
        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, self.id * PYDTNN_OPS_EVENTS + PYDTNN_OPS_FORWARD_IM2COL)
        y, idx_max = max_pool_2d_fwd_nchw_cython(x, self.kh, self.kw, self.vpadding, self.hpadding,
                                                 self.vstride, self.hstride, self.vdilation, self.hdilation)
        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, 0)
        if self.model.mode == TRAIN_MODE:
            self.idx_max = idx_max
        return y

    def _backward_nhwc_i2c(self, dy):
        if self.need_dx:
            dy_rows = np.zeros((np.prod(dy.shape), self.kh * self.kw), dtype=self.model.dtype)
            dy_rows[self.idx_max] = dy.flatten()
            self.model.tracer.emit_event(PYDTNN_OPS_EVENT, self.id * PYDTNN_OPS_EVENTS + PYDTNN_OPS_COMP_DX_COL2IM)
            dx = row2im_1ch_nhwc_cython(dy_rows, dy.shape[0], self.hi, self.wi, self.ci,
                                        self.kh, self.kw, self.vpadding, self.hpadding,
                                        self.vstride, self.hstride, self.vdilation, self.hdilation)
            self.model.tracer.emit_event(PYDTNN_OPS_EVENT, 0)
            dx = dx.reshape(-1, self.hi, self.wi, self.ci)
            return dx

    def _backward_nhwc_cython(self, dy):
        if self.need_dx:
            self.model.tracer.emit_event(PYDTNN_OPS_EVENT, self.id * PYDTNN_OPS_EVENTS + PYDTNN_OPS_COMP_DX_COL2IM)
            dx = max_pool_2d_bwd_nhwc_cython(dy, self.idx_max, dy.shape[0], self.hi, self.wi, self.ci,
                                             self.kh, self.kw, self.vpadding, self.hpadding,
                                             self.vstride, self.hstride, self.vdilation, self.hdilation)
            self.model.tracer.emit_event(PYDTNN_OPS_EVENT, 0)
            return dx

    def _backward_nchw_i2c(self, dy):
        if self.need_dx:
            dy_cols = np.zeros((self.kh * self.kw, np.prod(dy.shape)), dtype=self.model.dtype)
            dy_cols[self.idx_max] = dy.flatten()
            self.model.tracer.emit_event(PYDTNN_OPS_EVENT, self.id * PYDTNN_OPS_EVENTS + PYDTNN_OPS_COMP_DX_COL2IM)
            dx = col2im_1ch_nchw_cython(dy_cols, dy.shape[0], self.hi, self.wi, self.ci,
                                        self.kh, self.kw, self.vpadding, self.hpadding,
                                        self.vstride, self.hstride, self.vdilation, self.hdilation)
            self.model.tracer.emit_event(PYDTNN_OPS_EVENT, 0)
            dx = dx.reshape(-1, self.ci, self.hi, self.wi)
            return dx

    def _backward_nchw_cython(self, dy):
        if self.need_dx:
            self.model.tracer.emit_event(PYDTNN_OPS_EVENT, self.id * PYDTNN_OPS_EVENTS + PYDTNN_OPS_COMP_DX_COL2IM)
            dx = max_pool_2d_bwd_nchw_cython(dy, self.idx_max, dy.shape[0], self.hi, self.wi, self.ci,
                                             self.kh, self.kw, self.vpadding, self.hpadding,
                                             self.vstride, self.hstride, self.vdilation, self.hdilation)
            self.model.tracer.emit_event(PYDTNN_OPS_EVENT, 0)
            return dx

