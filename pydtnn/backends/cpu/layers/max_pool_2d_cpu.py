#
#  This file is part of Python Distributed Training of Neural Networks (PyDTNN)
#
#  Copyright (C) 2021 Universitat Jaume I
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
from pydtnn.cython_modules import im2col_cython, argmax_cython, col2im_cython
from pydtnn.layers import MaxPool2D
from pydtnn.model import TRAIN_MODE
from pydtnn.tracers import PYDTNN_OPS_EVENT, PYDTNN_OPS_EVENTS, PYDTNN_OPS_COMP_DX_COL2IM, PYDTNN_OPS_FORWARD_IM2COL


class MaxPool2DCPU(AbstractPool2DLayerCPU, MaxPool2D):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.maxids = None

    def forward(self, x):
        x_ = x.reshape(x.shape[0] * self.ci, 1, self.hi, self.wi)
        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, self.id * PYDTNN_OPS_EVENTS + PYDTNN_OPS_FORWARD_IM2COL)
        x_cols = im2col_cython(x_, self.kh, self.kw, self.vpadding, self.hpadding,
                               self.vstride, self.hstride)
        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, 0)

        # self.maxids = tuple([np.argmax(a_cols, axis=0), np.arange(a_cols.shape[1])])
        y, maxids = argmax_cython(x_cols, axis=0)

        if self.model.mode == TRAIN_MODE:
            self.maxids = maxids

        return y.reshape(x.shape[0], *self.shape)

    def backward(self, dy):
        if self.need_dx:
            dy_cols = np.zeros((self.kh * self.kw, np.prod(dy.shape)), dtype=self.model.dtype)
            dy_cols[self.maxids] = dy.flatten()
            self.model.tracer.emit_event(PYDTNN_OPS_EVENT, self.id * PYDTNN_OPS_EVENTS + PYDTNN_OPS_COMP_DX_COL2IM)
            dx = col2im_cython(dy_cols, dy.shape[0] * self.ci, 1, self.hi, self.wi,
                               self.kh, self.kw, self.vpadding, self.hpadding,
                               self.vstride, self.hstride)
            self.model.tracer.emit_event(PYDTNN_OPS_EVENT, 0)
            dx = dx.reshape(dy.shape[0], self.ci, self.hi, self.wi)
            return dx
