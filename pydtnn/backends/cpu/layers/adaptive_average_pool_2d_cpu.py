#
#  This file is part of Python Distributed Training of Neural Networks (PyDTNN)
#
#  Copyright (C) 2025 Universitat Jaume I
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
from pydtnn.utils import PYDTNN_TENSOR_FORMAT

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
        self.y: np.ndarray = None
        self.dx: np.ndarray = None
    # -- END __init__ -- #
        
    # Method from AbstractPool2DLayerCPU
    def initialize(self, prev_shape: tuple[int, int], need_dx:bool = True):
        # The objective is following lines is to override the AbstractPool2DLayer's initialize method, that is avoiding call to "super" since in that case AbstractPool2DLayer will be called eventually.
        AdaptiveAveragePool2D.initialize(self, prev_shape, need_dx)
        LayerCPU.initialize(self, prev_shape, need_dx)
        self.y = np.empty((self.model.batch_size, self.ci, self.ho, self.wo), dtype = self.model.dtype)
        self.dx = np.empty((self.model.batch_size, self.ci, self.hi, self.wi), dtype = self.model.dtype)

        if self.model.tensor_format == PYDTNN_TENSOR_FORMAT.NCHW:
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
    def forward(self, x: np.ndarray) -> np.ndarray:
        return self._forward(x)
    # --- END forward --- #

    @override    
    def backward(self, dy: np.ndarray) -> np.ndarray:
        return self._backward(dy)
    # --- END backward --- #

    # NOTE: I dont' know why, but if you try to set a variable with "_forward_pooling_not_needed", the value is None insted of the function.
    #   It's possible that the problem originates in "PromoteToBackendMixin"'s "__new__" method, but it's not sure the problem originates there.
    def _forward_pooling_not_needed(self, x:np.ndarray) -> np.ndarray:
        # If the output shape is the same as the input one, there is no need to make the pooling.
        return x
    # --- END _forward_pooling_not_needed --- #

    def _forward_nhwc_cython(self, x: np.ndarray) -> np.ndarray:
        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, self.id * PYDTNN_OPS_EVENTS + PYDTNN_OPS_EVENT_enum.FORWARD_ADP_AVG_POOL)
        adaptive_avg_pooling_fwd_nhwc_cython(x, self.y, self.ho, self.wo)
        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, PYDTNN_EVENT_FINISHED)        
        return self.y

    def _forward_nchw_cython(self, x: np.ndarray) -> np.ndarray:
        print(f"{x.shape=}")
        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, self.id * PYDTNN_OPS_EVENTS + PYDTNN_OPS_EVENT_enum.FORWARD_ADP_AVG_POOL)        
        adaptive_avg_pooling_fwd_nchw_cython(x, self.y, self.ho, self.wo)
        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, PYDTNN_EVENT_FINISHED)        
        return self.y

    def _backward_nhwc_cython(self, dy: np.ndarray) -> np.ndarray:
        if self.need_dx:
            self.model.tracer.emit_event(PYDTNN_OPS_EVENT, self.id * PYDTNN_OPS_EVENTS + PYDTNN_OPS_EVENT_enum.BACKWARD_ADP_AVG_POOL)
            adaptive_avg_pooling_bwd_nhwc_cython(dy, self.dx, self.hi, self.wi)
            self.model.tracer.emit_event(PYDTNN_OPS_EVENT, PYDTNN_EVENT_FINISHED)
            return self.dx
    # --- END _backward_nhwc_cython --- #
        
    def _backward_nchw_cython(self, dy: np.ndarray) -> np.ndarray:
        if self.need_dx:
            self.model.tracer.emit_event(PYDTNN_OPS_EVENT, self.id * PYDTNN_OPS_EVENTS + PYDTNN_OPS_EVENT_enum.BACKWARD_ADP_AVG_POOL)
            adaptive_avg_pooling_bwd_nchw_cython(dy, self.dx, self.hi, self.wi)
            self.model.tracer.emit_event(PYDTNN_OPS_EVENT, PYDTNN_EVENT_FINISHED)
            return self.dx
    # --- END _backward_nchw_cython --- #    
# --- END AdaptiveAveragePool2DCPU --- #
