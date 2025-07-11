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

from pydtnn.cython_modules import depthwise_conv_nchw_cython, add_cython, depthwise_conv_backward_nchw_cython, \
                                  depthwise_conv_nhwc_cython, add_cython, depthwise_conv_backward_nhwc_cython
from pydtnn.layers import Conv2D
from pydtnn.tracers import PYDTNN_OPS_EVENT, PYDTNN_OPS_EVENTS, PYDTNN_EVENT_FINISHED, PYDTNN_OPS_EVENT_enum

from pydtnn.utils.best_transpose_1023 import best_transpose_1023

import numpy as np

class DepthwiseVariant(Conv2D, ABC):
    # NOTE: Attributes defined in conv_2d_cpu.
    res:np.ndarray
    dx:np.ndarray
    dw:np.ndarray
    db:np.ndarray
    #---

    def _forward_depthwise_nhwc(self, x:np.ndarray) -> np.ndarray:
        """ Version of the forward that perform a depthwise convolution"""
        
        self.x = x
        res:np.ndarray = self.res[: x.shape[0], :]

        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, self.id * PYDTNN_OPS_EVENTS + PYDTNN_OPS_EVENT_enum.FORWARD_DEPTHWISE_CONV)
        depthwise_conv_nhwc_cython(x, self.weights, res, self.ho, self.wo,
                                   self.vpadding, self.hpadding,
                                   self.vstride, self.hstride, self.vdilation, self.hdilation)
        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, PYDTNN_EVENT_FINISHED)

        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, self.id * PYDTNN_OPS_EVENTS + PYDTNN_OPS_EVENT_enum.FORWARD_SUM_BIASES)
        add_cython(res.reshape((self.co, -1), copy=False), self.biases) if self.use_bias else res
        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, PYDTNN_EVENT_FINISHED)

        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, self.id * PYDTNN_OPS_EVENTS + PYDTNN_OPS_EVENT_enum.FORWARD_RESHAPE_Y)
        y = res.reshape((-1, self.ho, self.wo, self.co), copy=False)
        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, PYDTNN_EVENT_FINISHED)
        
        return y
    # --- END _forward_depthwise_nhwc --- #

    def _forward_depthwise_nchw(self, x:np.ndarray) -> np.ndarray:
        """ Version of the forward that perform a depthwise convolution"""
        self.x = x
        res = self.res[: x.shape[0], :]

        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, self.id * PYDTNN_OPS_EVENTS + PYDTNN_OPS_EVENT_enum.FORWARD_DEPTHWISE_CONV)
        depthwise_conv_nchw_cython(x, self.weights, res, self.ho, self.wo,
                                   self.vpadding, self.hpadding,
                                   self.vstride, self.hstride, self.vdilation, self.hdilation)
        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, PYDTNN_EVENT_FINISHED)

        _res = res.reshape((self.co, -1), copy=False)

        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, self.id * PYDTNN_OPS_EVENTS + PYDTNN_OPS_EVENT_enum.FORWARD_SUM_BIASES)
        add_cython(_res.reshape((self.co, -1), copy=False), self.biases) if self.use_bias else res
        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, PYDTNN_EVENT_FINISHED)

        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, self.id * PYDTNN_OPS_EVENTS + PYDTNN_OPS_EVENT_enum.FORWARD_RESHAPE_Y)
        y = best_transpose_1023(res.reshape((self.co, -1, self.ho, self.wo), copy=False))
        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, PYDTNN_EVENT_FINISHED)
        
        return y
    # --- END _forward_depthwise_nchw --- #

    def _backward_depthwise_nhwc(self, dy:np.ndarray) -> np.ndarray:
        
        dx = self.dx[: dy.shape[0], :]

        depthwise_conv_backward_nhwc_cython(dy, self.x, self.weights,
                                            dx, self.dw,
                                            self.vpadding, self.hpadding,
                                            self.vstride, self.hstride,
                                            self.vdilation, self.hdilation)

        if self.use_bias:
            self.model.tracer.emit_event(PYDTNN_OPS_EVENT, self.id * PYDTNN_OPS_EVENTS + PYDTNN_OPS_EVENT_enum.BACKWARD_SUM_BIASES)
            np.sum(dy, axis=(0, 1, 2), out=self.db)
            self.model.tracer.emit_event(PYDTNN_OPS_EVENT, PYDTNN_EVENT_FINISHED)

        return dx
    # --- END _backward_depthwise_nhwc --- #

    def _backward_depthwise_nchw(self, dy:np.ndarray) -> np.ndarray:
        
        dx = self.dx[: dy.shape[0], :]
    
        depthwise_conv_backward_nchw_cython(dy, self.x, self.weights,
                                            dx, self.dw,
                                            self.vpadding, self.hpadding,
                                            self.vstride, self.hstride,
                                            self.vdilation, self.hdilation)

        if self.use_bias:
            self.model.tracer.emit_event(PYDTNN_OPS_EVENT, self.id * PYDTNN_OPS_EVENTS + PYDTNN_OPS_EVENT_enum.BACKWARD_SUM_BIASES)
            np.sum(dy, axis=(0, 2, 3), out=self.db)
            self.model.tracer.emit_event(PYDTNN_OPS_EVENT, PYDTNN_EVENT_FINISHED)

        return dx
    # --- END _backward_depthwise_nchw --- #
