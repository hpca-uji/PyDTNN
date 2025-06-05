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
from typing import override

from abc import ABC
from pydtnn.layers import AdaptiveAveragePool2D
from pydtnn.backends.cpu.layers import LayerCPU

# Imports for the method from AbstractPool2DLayerCPU
from pydtnn.utils import PYDTNN_TENSOR_FORMAT_NCHW

# Imports for the methods from AveragePool2DCPU
from pydtnn.cython_modules import im2row_1ch_nhwc_cython, row2im_1ch_nhwc_cython, \
                                  im2col_1ch_nchw_cython, col2im_1ch_nchw_cython, \
                                  adaptive_avg_pooling_fwd_nchw_cython, adaptive_avg_pooling_bwd_nchw_cython, \
                                  adaptive_avg_pooling_fwd_nhwc_cython, adaptive_avg_pooling_bwd_nhwc_cython
from pydtnn.tracers import PYDTNN_OPS_EVENT, PYDTNN_OPS_EVENTS, PYDTNN_EVENT_FINISHED, PYDTNN_OPS_EVENT_enum    
import numpy as np

class AdaptiveAveragePool2DCPU(AdaptiveAveragePool2D, LayerCPU, ABC):
    # The backend is almost the same as a AveragePool2D layer.

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    # -- END __init__ -- #
        
    # Method from AbstractPool2DLayerCPU
    def initialize(self, prev_shape: tuple[int, int], need_dx:bool = True):
        # The objective is following lines is to override the AbstractPool2DLayer's initialize method, that is avoiding call to "super" since in that case AbstractPool2DLayer will be called eventually.
        AdaptiveAveragePool2D.initialize(self, prev_shape, need_dx)
        LayerCPU.initialize(self, prev_shape, need_dx)

        if self.model.tensor_format == PYDTNN_TENSOR_FORMAT_NCHW:
            self._forward = self._forward_nchw_cython
            self._backward = self._backward_nchw_cython
            # I2C-based implementations have been temporarily discarded
            # setattr(self, "forward", self._forward_nchw_i2c)
            # setattr(self, "backward", self._backward_nchw_i2c)
        else: # Assuming PYDTNN_TENSOR_FORMAT_NHWC
            self._forward = self._forward_nhwc_cython
            self._backward = self._backward_nhwc_cython
            # I2C-based implementations have been temporarily discarded
            # setattr(self, "forward", self._forward_nhwc_i2c)
            # setattr(self, "backward", self._backward_nhwc_i2c)

        if self.pooling_not_needed:
            #self._forward = self._forward_pooling_not_needed # NOTE: See "self._forward_pooling_not_needed"
            self._forward = (lambda x: x)
        #else: Nothing special.

    # -- END initialize -- #
    
    @override
    def forward(self, x):
        return self._forward(x)
    # --- END forward --- #

    @override    
    def backward(self, dy):
        return self._backward(dy)
    # --- END backward --- #

    # NOTE: I dont' know why, but if you try to set a variable with "_forward_pooling_not_needed", the value is None insted of the function.
    #   It's possible that the problem originates in "PromoteToBackendMixin"'s "__new__" method, but it's not sure the problem originates there.
    def _forward_pooling_not_needed(self, x:np.ndarray) -> np.ndarray:
        # If the output shape is the same as the input one, there is no need to make the pooling.
        return x
    # --- END _forward_pooling_not_needed --- #

    # Methods from AveragePool2DCPU
    def _forward_nhwc_i2c(self, x):
            # TODO: Implement this or adapt it to call the "_forward_nchw_cython" until is implemented
        raise NotImplementedError("_forward_nhwc_i2c not implemented for Adaptive Average Pooling!")
        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, self.id * PYDTNN_OPS_EVENTS + PYDTNN_OPS_EVENT_enum.FORWARD_IM2COL)
        x_rows = im2row_1ch_nhwc_cython(x, self.kh, self.kw, self.vpadding, self.hpadding,
                                        self.vstride, self.hstride, self.vdilation, self.hdilation)
        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, PYDTNN_EVENT_FINISHED)
        y = np.mean(x_rows, axis=1)
        return y.reshape(-1, self.ho, self.wo, self.co)

    def _forward_nhwc_cython(self, x):
        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, self.id * PYDTNN_OPS_EVENTS + PYDTNN_OPS_EVENT_enum.FORWARD_ADP_AVG_POOL)
        y = adaptive_avg_pooling_fwd_nhwc_cython(x, self.kh, self.kw, self.vpadding, self.hpadding,
                                        self.vstride, self.hstride, self.vdilation, self.hdilation)
        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, PYDTNN_EVENT_FINISHED)        
        return y

    def _forward_nchw_i2c(self, x):
        # TODO: Implement this or adapt it to call the "_forward_nchw_cython" until is implemented
        raise NotImplementedError("_forward_nchw_i2c not implemented for Adaptive Average Pooling!")
        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, self.id * PYDTNN_OPS_EVENTS + PYDTNN_OPS_EVENT_enum.FORWARD_IM2COL)
        x_cols = im2col_1ch_nchw_cython(x, self.kh, self.kw, self.vpadding, self.hpadding,
                                        self.vstride, self.hstride, self.vdilation, self.hdilation)
        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, PYDTNN_EVENT_FINISHED)
        y = np.mean(x_cols, axis=0)
        return y.reshape(-1, self.co, self.ho, self.wo)

    def _forward_nchw_cython(self, x):
        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, self.id * PYDTNN_OPS_EVENTS + PYDTNN_OPS_EVENT_enum.FORWARD_ADP_AVG_POOL)        
        y = adaptive_avg_pooling_fwd_nchw_cython(x, self.ho, self.wo)
        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, PYDTNN_EVENT_FINISHED)        
        return y

    def _backward_nhwc_i2c(self, dy):
        if self.need_dx:
            pool_size = np.prod(self.pool_shape)
            dy_rows = np.tile(dy.reshape(-1, 1) / pool_size, (1, pool_size))
            self.model.tracer.emit_event(PYDTNN_OPS_EVENT, self.id * PYDTNN_OPS_EVENTS + PYDTNN_OPS_EVENT_enum.COMP_DX_COL2IM)
            dx = row2im_1ch_nhwc_cython(dy_rows, dy.shape[0], self.hi, self.wi, self.ci,
                                        self.kh, self.kw, self.vpadding, self.hpadding,
                                        self.vstride, self.hstride, self.vdilation, self.hdilation)
            self.model.tracer.emit_event(PYDTNN_OPS_EVENT, PYDTNN_EVENT_FINISHED)
            dx = dx.reshape(-1, self.hi, self.wi, self.ci)
            return dx
    # END Methods from AveragePool2DCPU
    
    def _backward_nchw_i2c(self, dy):
        if self.need_dx:
            pool_size = np.prod(self.pool_shape)
            dy_cols = np.tile(dy.flatten() / pool_size, (pool_size, 1))
            self.model.tracer.emit_event(PYDTNN_OPS_EVENT, self.id * PYDTNN_OPS_EVENTS + PYDTNN_OPS_EVENT_enum.COMP_DX_COL2IM)
            dx = col2im_1ch_nchw_cython(dy_cols, dy.shape[0], self.hi, self.wi, self.ci,
                                        self.kh, self.kw, self.vpadding, self.hpadding,
                                        self.vstride, self.hstride, self.vdilation, self.hdilation)
            self.model.tracer.emit_event(PYDTNN_OPS_EVENT, PYDTNN_EVENT_FINISHED)
            dx = dx.reshape(-1, self.ci, self.hi, self.wi)
            return dx

    def _backward_nhwc_cython(self, dy: np.ndarray) -> np.ndarray:
        if self.need_dx:
            self.model.tracer.emit_event(PYDTNN_OPS_EVENT, self.id * PYDTNN_OPS_EVENTS + PYDTNN_OPS_EVENT_enum.BACKWARD_ADP_AVG_POOL)
            dx = adaptive_avg_pooling_bwd_nhwc_cython(dy, self.hi, self.wi)
            self.model.tracer.emit_event(PYDTNN_OPS_EVENT, PYDTNN_EVENT_FINISHED)
            return dx
    # --- END _backward_nhwc_cython --- #
        
    def _backward_nchw_cython(self, dy: np.ndarray) -> np.ndarray:
        if self.need_dx:
            self.model.tracer.emit_event(PYDTNN_OPS_EVENT, self.id * PYDTNN_OPS_EVENTS + PYDTNN_OPS_EVENT_enum.BACKWARD_ADP_AVG_POOL)
            dx = adaptive_avg_pooling_bwd_nchw_cython(dy, self.hi, self.wi)
            self.model.tracer.emit_event(PYDTNN_OPS_EVENT, PYDTNN_EVENT_FINISHED)
            return dx
    # --- END _backward_nchw_cython --- #    
# --- END AdaptiveAveragePool2DCPU --- #
