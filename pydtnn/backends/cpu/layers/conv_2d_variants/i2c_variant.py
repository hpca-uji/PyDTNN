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

from abc import ABC

import numpy as np

from pydtnn.cython_modules import im2row_nhwc_cython, add_nhwc_cython, im2col_nchw_cython, add_nchw_cython, \
    row2im_nhwc_cython, col2im_nchw_cython
from pydtnn.layers import Conv2D
from pydtnn.model import TRAIN_MODE
from pydtnn.tracers import PYDTNN_OPS_EVENT, PYDTNN_OPS_EVENTS, PYDTNN_EVENT_FINISHED, PYDTNN_OPS_EVENT_enum

from pydtnn.utils.best_transpose_1023 import best_transpose_1023


class I2CVariant(Conv2D, ABC):

    def _forward_i2c_nhwc(self, x: np.ndarray) -> np.ndarray:
        """Version of the forward function that uses im2col and matmul"""

        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, self.id * PYDTNN_OPS_EVENTS + PYDTNN_OPS_EVENT_enum.FORWARD_IM2COL)
        x_rows = im2row_nhwc_cython(x, self.kh, self.kw, self.vpadding, self.hpadding,
                                    self.vstride, self.hstride, self.vdilation, self.hdilation)
        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, PYDTNN_EVENT_FINISHED)

        if self.model.mode == TRAIN_MODE:
            self.x_rows = x_rows

        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, self.id * PYDTNN_OPS_EVENTS + PYDTNN_OPS_EVENT_enum.FORWARD_RESHAPE_W)
        w_cols = self.weights.reshape((-1, self.co), copy=False)
        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, PYDTNN_EVENT_FINISHED)

        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, self.id * PYDTNN_OPS_EVENTS + PYDTNN_OPS_EVENT_enum.FORWARD_MATMUL)
        res = self.model.matmul(x_rows, w_cols)
        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, PYDTNN_EVENT_FINISHED)

        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, self.id * PYDTNN_OPS_EVENTS + PYDTNN_OPS_EVENT_enum.FORWARD_SUM_BIASES)
        y = add_nhwc_cython(res, self.biases) if self.use_bias else res
        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, PYDTNN_EVENT_FINISHED)

        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, self.id * PYDTNN_OPS_EVENTS + PYDTNN_OPS_EVENT_enum.FORWARD_RESHAPE_Y)
        y = y.reshape((-1, self.ho, self.wo, self.co), copy=False)
        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, PYDTNN_EVENT_FINISHED)
        return y

    def _forward_i2c_nchw(self, x: np.ndarray) -> np.ndarray:
        """Version of the forward function that uses im2col and matmul"""

        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, self.id * PYDTNN_OPS_EVENTS + PYDTNN_OPS_EVENT_enum.FORWARD_IM2COL)
        x_cols: np.ndarray = im2col_nchw_cython(x, self.kh, self.kw, self.vpadding, self.hpadding,
                                                self.vstride, self.hstride, self.vdilation, self.hdilation)
        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, PYDTNN_EVENT_FINISHED)

        if self.model.mode == TRAIN_MODE:
            self.x_cols = x_cols

        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, self.id * PYDTNN_OPS_EVENTS + PYDTNN_OPS_EVENT_enum.FORWARD_RESHAPE_W)
        w_cols = self.weights.reshape((self.co, -1), copy=False)
        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, PYDTNN_EVENT_FINISHED)

        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, self.id * PYDTNN_OPS_EVENTS + PYDTNN_OPS_EVENT_enum.FORWARD_MATMUL)
        res = self.model.matmul(w_cols, x_cols)
        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, PYDTNN_EVENT_FINISHED)

        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, self.id * PYDTNN_OPS_EVENTS + PYDTNN_OPS_EVENT_enum.FORWARD_SUM_BIASES)
        y = add_nchw_cython(res, self.biases) if self.use_bias else res
        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, PYDTNN_EVENT_FINISHED)

        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, self.id * PYDTNN_OPS_EVENTS + PYDTNN_OPS_EVENT_enum.FORWARD_RESHAPE_Y)
        y:np.ndarray = best_transpose_1023(y.reshape((self.co, -1, self.ho, self.wo), copy=False))
        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, PYDTNN_EVENT_FINISHED)

        return y

    def _backward_i2c_nhwc(self, dy: np.ndarray) -> np.ndarray | None:
        """Version of the backward function that uses im2col and matmul"""
        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, self.id * PYDTNN_OPS_EVENTS + PYDTNN_OPS_EVENT_enum.BACKWARD_TRANSPOSE_DY)
        dy_rows = dy.reshape((-1, self.co), copy=False)
        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, PYDTNN_EVENT_FINISHED)

        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, self.id * PYDTNN_OPS_EVENTS + PYDTNN_OPS_EVENT_enum.COMP_DW_MATMUL)
        res = self.model.matmul(self.x_rows.T, dy_rows)
        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, PYDTNN_EVENT_FINISHED)

        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, self.id * PYDTNN_OPS_EVENTS + PYDTNN_OPS_EVENT_enum.BACKWARD_RESHAPE_DW)
        self.dw = res.reshape(self.weights.shape)
        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, PYDTNN_EVENT_FINISHED)

        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, self.id * PYDTNN_OPS_EVENTS + PYDTNN_OPS_EVENT_enum.BACKWARD_SUM_BIASES)
        if self.use_bias:
            self.db = np.sum(dy, axis=(0, 1, 2))
        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, PYDTNN_EVENT_FINISHED)

        if self.need_dx:
            self.model.tracer.emit_event(PYDTNN_OPS_EVENT, self.id * PYDTNN_OPS_EVENTS + PYDTNN_OPS_EVENT_enum.BACKWARD_TRANSPOSE_W)
            w_rows = self.weights.reshape((-1, self.co), copy=False)
            self.model.tracer.emit_event(PYDTNN_OPS_EVENT, PYDTNN_EVENT_FINISHED)

            self.model.tracer.emit_event(PYDTNN_OPS_EVENT, self.id * PYDTNN_OPS_EVENTS + PYDTNN_OPS_EVENT_enum.COMP_DX_MATMUL)
            res = self.model.matmul(dy_rows, w_rows.T)
            self.model.tracer.emit_event(PYDTNN_OPS_EVENT, PYDTNN_EVENT_FINISHED)

            self.model.tracer.emit_event(PYDTNN_OPS_EVENT, self.id * PYDTNN_OPS_EVENTS + PYDTNN_OPS_EVENT_enum.COMP_DX_COL2IM)
            dx = row2im_nhwc_cython(res, dy.shape[0], self.hi, self.wi, self.ci,
                                    self.kh, self.kw, self.vpadding, self.hpadding,
                                    self.vstride, self.hstride, self.vdilation, self.hdilation)
            self.model.tracer.emit_event(PYDTNN_OPS_EVENT, PYDTNN_EVENT_FINISHED)
            return dx

    def _backward_i2c_nchw(self, dy: np.ndarray) -> np.ndarray | None:
        """Version of the backward function that uses im2col and matmul"""
        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, self.id * PYDTNN_OPS_EVENTS + PYDTNN_OPS_EVENT_enum.BACKWARD_TRANSPOSE_DY)
        dy_cols:np.ndarray = best_transpose_1023(dy).reshape((self.co, -1), copy=False)
        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, PYDTNN_EVENT_FINISHED)

        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, self.id * PYDTNN_OPS_EVENTS + PYDTNN_OPS_EVENT_enum.COMP_DW_MATMUL)
        res = self.model.matmul(dy_cols, self.x_cols.T)
        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, PYDTNN_EVENT_FINISHED)

        del self.x_cols
        self.x_cols = None

        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, self.id * PYDTNN_OPS_EVENTS + PYDTNN_OPS_EVENT_enum.BACKWARD_RESHAPE_DW)
        self.dw = res.reshape(self.weights.shape, copy=False)
        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, PYDTNN_EVENT_FINISHED)

        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, self.id * PYDTNN_OPS_EVENTS + PYDTNN_OPS_EVENT_enum.BACKWARD_SUM_BIASES)
        if self.use_bias:
            self.db = np.sum(dy, axis=(0, 2, 3))
        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, PYDTNN_EVENT_FINISHED)

        if self.need_dx:
            self.model.tracer.emit_event(PYDTNN_OPS_EVENT,
                                         self.id * PYDTNN_OPS_EVENTS + PYDTNN_OPS_EVENT_enum.BACKWARD_TRANSPOSE_W)
            w_cols = self.weights.reshape((self.co, -1), copy=False).T
            self.model.tracer.emit_event(PYDTNN_OPS_EVENT, PYDTNN_EVENT_FINISHED)

            self.model.tracer.emit_event(PYDTNN_OPS_EVENT, self.id * PYDTNN_OPS_EVENTS + PYDTNN_OPS_EVENT_enum.COMP_DX_MATMUL)
            res = self.model.matmul(w_cols, dy_cols)
            self.model.tracer.emit_event(PYDTNN_OPS_EVENT, PYDTNN_EVENT_FINISHED)

            self.model.tracer.emit_event(PYDTNN_OPS_EVENT, self.id * PYDTNN_OPS_EVENTS + PYDTNN_OPS_EVENT_enum.COMP_DX_COL2IM)
            dx:np.ndarray = col2im_nchw_cython(res, dy.shape[0], self.ci, self.hi, self.wi,
                                               self.kh, self.kw, self.vpadding, self.hpadding,
                                               self.vstride, self.hstride, self.vdilation, self.hdilation)
            self.model.tracer.emit_event(PYDTNN_OPS_EVENT, PYDTNN_EVENT_FINISHED)
            return dx
