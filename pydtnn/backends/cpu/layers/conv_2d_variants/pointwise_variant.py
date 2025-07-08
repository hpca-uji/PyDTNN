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

from abc import ABC

import numpy as np

from pydtnn.layers import Conv2D
from pydtnn.tracers import PYDTNN_OPS_EVENT, PYDTNN_OPS_EVENTS, PYDTNN_OPS_EVENT_enum, PYDTNN_EVENT_FINISHED
from pydtnn.utils.best_transpose_0231 import best_transpose_0231
from pydtnn.utils.best_transpose_0312 import best_transpose_0312
from pydtnn.model import TRAIN_MODE


class PointwiseVariant(Conv2D, ABC):

    def _forward_pointwise_nhwc(self, x: np.ndarray) -> np.ndarray:
        if self.model.mode == TRAIN_MODE:
            self._x:np.ndarray = x

        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, self.id * PYDTNN_OPS_EVENTS + PYDTNN_OPS_EVENT_enum.FORWARD_POINTWISE_CONV)
        y = np.linalg.matmul(x, self.weights)
        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, PYDTNN_EVENT_FINISHED)

        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, self.id * PYDTNN_OPS_EVENTS + PYDTNN_OPS_EVENT_enum.FORWARD_SUM_BIASES)
        if self.use_bias:
            # TODO: POSIBLE MEJORA
            y += self.biases.reshape((1, 1, 1, self.co), copy=False)
        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, PYDTNN_EVENT_FINISHED)
        return y
    # --- END _forward_pointwise_nhwc --- #

    def _forward_pointwise_nchw(self, x: np.ndarray) -> np.ndarray:        

        if self.model.mode == TRAIN_MODE:
            self._x:np.ndarray = x

        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, self.id * PYDTNN_OPS_EVENTS + PYDTNN_OPS_EVENT_enum.FORWARD_POINTWISE_CONV)
        y = np.linalg.matmul(best_transpose_0231(x), self.weights.T)
        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, PYDTNN_EVENT_FINISHED)

        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, self.id * PYDTNN_OPS_EVENTS + PYDTNN_OPS_EVENT_enum.FORWARD_TRANSPOSE_Y)
        y = best_transpose_0312(y)
        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, PYDTNN_EVENT_FINISHED)

        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, self.id * PYDTNN_OPS_EVENTS + PYDTNN_OPS_EVENT_enum.FORWARD_SUM_BIASES)
        if self.use_bias:
            # TODO: POSIBLE MEJORA
            y += self.biases.reshape((1, self.co, 1, 1), copy=False)
        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, PYDTNN_EVENT_FINISHED)
        return y
    # --- END _forward_pointwise_nchw --- #

    def _backward_pointwise_nhwc(self, dy: np.ndarray) -> np.ndarray:

        _n, _h, _w, _c = dy.shape
        _dim = _n * _h * _w
        x_shape = self._x.shape

        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, self.id * PYDTNN_OPS_EVENTS + PYDTNN_OPS_EVENT_enum.BACKWARD_TRANSPOSE_DY)
        # NOTE: "n", "h" and "w" must be the same, "c" is the only dimension that changes
        reshaped_dy = dy.reshape((_dim, _c), copy = False)
        self._x = self._x.reshape((-1, _dim), copy = False)
        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, PYDTNN_EVENT_FINISHED)

        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, self.id * PYDTNN_OPS_EVENTS + PYDTNN_OPS_EVENT_enum.COMP_DW_MATMUL)
        res:np.ndarray = np.linalg.matmul(self._x, reshaped_dy)
        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, PYDTNN_EVENT_FINISHED)

        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, self.id * PYDTNN_OPS_EVENTS + PYDTNN_OPS_EVENT_enum.BACKWARD_RESHAPE_DW)
        self.dw = res.reshape((self.weights.shape))
        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, PYDTNN_EVENT_FINISHED)

        if self.use_bias:
            self.model.tracer.emit_event(PYDTNN_OPS_EVENT, self.id * PYDTNN_OPS_EVENTS + PYDTNN_OPS_EVENT_enum.BACKWARD_SUM_BIASES)
            self.db:np.ndarray = np.sum(dy, axis=(0, 1, 2)).reshape((self.co,))
            self.model.tracer.emit_event(PYDTNN_OPS_EVENT, PYDTNN_EVENT_FINISHED)

        self.model.tracer.emit_event(PYDTNN_OPS_EVENT,self.id * PYDTNN_OPS_EVENTS + PYDTNN_OPS_EVENT_enum.BACKWARD_TRANSPOSE_W)    
        w = self.weights.reshape((self.co, -1)).T
        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, PYDTNN_EVENT_FINISHED)

        reshaped_dy:np.ndarray = dy.reshape((self.co, -1))

        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, self.id * PYDTNN_OPS_EVENTS + PYDTNN_OPS_EVENT_enum.COMP_DX_MATMUL)
        dx:np.ndarray = np.linalg.matmul(w, reshaped_dy)
        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, PYDTNN_EVENT_FINISHED)

        # NOTE: Remember, dx must have the forward's input shape.
        dx = dx.reshape(x_shape, copy=False)
        
        return dx
    # --- END _backward_pointwise_nhwc --- #

    def _backward_pointwise_nchw(self, dy: np.ndarray) -> np.ndarray:
        
        _n, _c, _h, _w = dy.shape
        _dim = _n * _h * _w
        x_shape = self._x.shape

        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, self.id * PYDTNN_OPS_EVENTS + PYDTNN_OPS_EVENT_enum.BACKWARD_TRANSPOSE_DY)
        # NOTE: "n", "h" and "w" must be the same, "c" is the only dimension that changes
        reshaped_dy = dy.reshape((_dim, _c))
        self._x = self._x.reshape((-1, _dim))
        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, PYDTNN_EVENT_FINISHED)

        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, self.id * PYDTNN_OPS_EVENTS + PYDTNN_OPS_EVENT_enum.COMP_DW_MATMUL)
        res:np.ndarray = np.linalg.matmul(self._x, reshaped_dy)
        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, PYDTNN_EVENT_FINISHED)

        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, self.id * PYDTNN_OPS_EVENTS + PYDTNN_OPS_EVENT_enum.BACKWARD_RESHAPE_DW)
        self.dw = res.reshape((self.weights.shape))
        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, PYDTNN_EVENT_FINISHED)

        if self.use_bias:
            self.model.tracer.emit_event(PYDTNN_OPS_EVENT, self.id * PYDTNN_OPS_EVENTS + PYDTNN_OPS_EVENT_enum.BACKWARD_SUM_BIASES)
            self.db:np.ndarray = np.sum(dy, axis=(0, 2, 3)).reshape((self.co,))
            self.model.tracer.emit_event(PYDTNN_OPS_EVENT, PYDTNN_EVENT_FINISHED)

        self.model.tracer.emit_event(PYDTNN_OPS_EVENT,self.id * PYDTNN_OPS_EVENTS + PYDTNN_OPS_EVENT_enum.BACKWARD_TRANSPOSE_W)    
        w = self.weights.reshape((self.co, -1), copy=False).T
        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, PYDTNN_EVENT_FINISHED)

        reshaped_dy:np.ndarray = dy.reshape((self.co, -1))

        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, self.id * PYDTNN_OPS_EVENTS + PYDTNN_OPS_EVENT_enum.COMP_DX_MATMUL)
        dx:np.ndarray = np.linalg.matmul(w, reshaped_dy)
        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, PYDTNN_EVENT_FINISHED)

        # NOTE: Remember, dx must have the same shape as forward input's shape.
        dx = dx.reshape((x_shape), copy=False)
        
        return dx
    # --- END _backward_pointwise_nchw --- #
