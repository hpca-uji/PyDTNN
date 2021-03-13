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
#  with this program.  If not, see <https://www.gnu.org/licenses/>.
#

from .layer import Layer
from ..cython_modules import im2col_cython, argmax_cython, col2im_cython
from ..model import TRAIN_MODE
from ..performance_models import *
from ..tracers import PYDTNN_OPS_EVENT, PYDTNN_OPS_EVENTS, PYDTNN_OPS_COMP_DX_COL2IM, PYDTNN_OPS_FORWARD_IM2COL


class MaxPool2D(Layer):

    def __init__(self, pool_shape=(2, 2), padding=0, stride=1):
        super(MaxPool2D, self).__init__()
        self.pool_shape = pool_shape
        self.padding = padding
        self.stride = stride
        self.vpadding, self.hpadding = (padding, padding) if isinstance(padding, int) else padding
        self.vstride, self.hstride = (stride, stride) if isinstance(stride, int) else stride
        # The next attributes will be initialized later
        self.ci = self.hi = self.wi = self.kh = self.kw = self.ho = self.wo = self.co = self.n = 0
        self.maxids = None

    def initialize(self, prev_shape, need_dx=True, x=None):
        super().initialize(prev_shape, need_dx)
        self.ci, self.hi, self.wi = prev_shape
        if self.pool_shape[0] == 0:
            self.pool_shape = (self.hi, self.pool_shape[1])
        if self.pool_shape[1] == 0:
            self.pool_shape = (self.pool_shape[0], self.wi)
        self.kh, self.kw = self.pool_shape
        self.ho = (self.hi + 2 * self.vpadding - self.kh) // self.vstride + 1
        self.wo = (self.wi + 2 * self.hpadding - self.kw) // self.hstride + 1
        self.co = self.ci
        self.shape = (self.co, self.ho, self.wo)
        self.n = np.prod(self.shape)

        self.fwd_time = \
            im2col_time(m=(self.kh * self.kw), n=(self.model.batch_size * self.ho * self.wo * self.ci),
                        cpu_speed=self.model.cpu_speed, memory_bw=self.model.memory_bw,
                        dtype=self.model.dtype)
        self.bwd_time = \
            col2im_time(m=(self.kh * self.kw), n=(self.model.batch_size * self.ho * self.wo * self.ci),
                        cpu_speed=self.model.cpu_speed, memory_bw=self.model.memory_bw,
                        dtype=self.model.dtype) if need_dx else 0

    def show(self, attrs=""):
        super().show("|{:^19s}|{:^24s}|".format(str(self.pool_shape),
                                                f"padd=({self.vpadding},{self.hpadding}), "
                                                f"stride=({self.vstride},{self.hstride})"))

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
