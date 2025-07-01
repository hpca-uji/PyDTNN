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

from pydtnn.cython_modules import depthwise_conv_nchw_cython, add_nchw_cython, depthwise_conv_backward_nchw_cython, \
                                  depthwise_conv_nhwc_cython, add_nhwc_cython, depthwise_conv_backward_nhwc_cython
from pydtnn.layers import Conv2D
from pydtnn.tracers import PYDTNN_OPS_EVENT, PYDTNN_OPS_EVENTS, PYDTNN_EVENT_FINISHED, PYDTNN_OPS_EVENT_enum

from pydtnn.utils.best_transpose_1023 import best_transpose_1023

import numpy as np

class DepthwiseVariant(Conv2D, ABC):

    def _forward_depthwise_nhwc(self, x:np.ndarray) -> np.ndarray:
        """ Version of the forward that perform a depthwise convolution"""

        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, self.id * PYDTNN_OPS_EVENTS + PYDTNN_OPS_EVENT_enum.FORWARD_DEPTHWISE_CONV)
        res:np.ndarray = depthwise_conv_nhwc_cython(x, self.weights, self.vpadding, self.hpadding,
                                                    self.vstride, self.hstride, self.vdilation, self.hdilation)
        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, PYDTNN_EVENT_FINISHED)

        if self.need_dx:
            self.x = x

        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, self.id * PYDTNN_OPS_EVENTS + PYDTNN_OPS_EVENT_enum.FORWARD_SUM_BIASES)
        add_nhwc_cython(res.reshape((self.co, -1), copy=False), self.biases) if self.use_bias else res
        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, PYDTNN_EVENT_FINISHED)

        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, self.id * PYDTNN_OPS_EVENTS + PYDTNN_OPS_EVENT_enum.FORWARD_RESHAPE_Y)
        y = res.reshape((-1, self.ho, self.wo, self.co), copy=False)
        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, PYDTNN_EVENT_FINISHED)

        return y
    # --- END _forward_depthwise_nhwc --- #

    def _forward_depthwise_nchw(self, x:np.ndarray) -> np.ndarray:
        """ Version of the forward that perform a depthwise convolution"""

        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, self.id * PYDTNN_OPS_EVENTS + PYDTNN_OPS_EVENT_enum.FORWARD_DEPTHWISE_CONV)
        res:np.ndarray = depthwise_conv_nchw_cython(x, self.weights, self.vpadding, self.hpadding,
                                                    self.vstride, self.hstride, self.vdilation, self.hdilation)
        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, PYDTNN_EVENT_FINISHED)

        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, self.id * PYDTNN_OPS_EVENTS + PYDTNN_OPS_EVENT_enum.FORWARD_SUM_BIASES)
        add_nchw_cython(res, self.biases) if self.use_bias else res
        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, PYDTNN_EVENT_FINISHED)

        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, self.id * PYDTNN_OPS_EVENTS + PYDTNN_OPS_EVENT_enum.FORWARD_RESHAPE_Y)
        y = best_transpose_1023(res.reshape((self.co, -1, self.ho, self.wo), copy=False))
        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, PYDTNN_EVENT_FINISHED)
        return y
    # --- END _forward_depthwise_nchw --- #

    def _backward_depthwise_nhwc(self, dy:np.ndarray) -> np.ndarray | None:
        
        #np.ndarray dx, self.dw
        dx, self.dw = depthwise_conv_backward_nhwc_cython(dy, self.x, self.weights,
                                                          self.vpadding, self.hpadding,
                                                          self.vstride, self.hstride,
                                                          self.vdilation, self.hdilation)

        if self.use_bias:
            self.model.tracer.emit_event(PYDTNN_OPS_EVENT, self.id * PYDTNN_OPS_EVENTS + PYDTNN_OPS_EVENT_enum.BACKWARD_SUM_BIASES)
            self.db:np.ndarray = np.sum(dy, axis=(0, 1, 2)).reshape((self.co,), copy=False)
            self.model.tracer.emit_event(PYDTNN_OPS_EVENT, PYDTNN_EVENT_FINISHED)

        if self.need_dx:
            return dx
    # --- END _backward_depthwise_nhwc --- #

    def _backward_depthwise_nchw(self, dy:np.ndarray) -> np.ndarray | None:
        
        #np.ndarray dx, self.dw
        dx, self.dw = depthwise_conv_backward_nchw_cython(dy, self.weights, self.x,
                                                          self.vpadding, self.hpadding,
                                                          self.vstride, self.hstride,
                                                          self.vdilation, self.hdilation)

        if self.use_bias:
            self.model.tracer.emit_event(PYDTNN_OPS_EVENT, self.id * PYDTNN_OPS_EVENTS + PYDTNN_OPS_EVENT_enum.BACKWARD_SUM_BIASES)
            self.db:np.ndarray = np.sum(dy, axis=(0, 2, 3)).reshape((self.co,), copy=False)
            self.model.tracer.emit_event(PYDTNN_OPS_EVENT, PYDTNN_EVENT_FINISHED)

        if self.need_dx:
            return dx
    # --- END _backward_depthwise_nchw --- #
