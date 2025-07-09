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
from pydtnn.cython_modules import adaptive_avg_pooling_fwd_nchw_cython, adaptive_avg_pooling_bwd_nchw_cython, \
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
    def initialize(self, prev_shape: tuple[int, int]):
        # The objective is following lines is to override the AbstractPool2DLayer's initialize method, that is avoiding call to "super" since in that case AbstractPool2DLayer will be called eventually.
        AdaptiveAveragePool2D.initialize(self, prev_shape)
        LayerCPU.initialize(self, prev_shape)

        if self.model.tensor_format == PYDTNN_TENSOR_FORMAT.NCHW:
            self.y = np.empty((self.model.batch_size, self.co, self.ho, self.wo), dtype = self.model.dtype)
            self.dx = np.empty((self.model.batch_size, self.ci, self.hi, self.wi), dtype = self.model.dtype)
        
            self._forward = self._forward_nchw_cython
            self._backward = self._backward_nchw_cython
        else: # Assuming PYDTNN_TENSOR_FORMAT_NHWC
            self.y = np.empty((self.model.batch_size, self.ho, self.wo, self.co), dtype = self.model.dtype)
            self.dx = np.empty((self.model.batch_size, self.hi, self.wi, self.ci), dtype = self.model.dtype)

            self._forward = self._forward_nhwc_cython
            self._backward = self._backward_nhwc_cython

        if self.pooling_not_needed:
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

    def _forward_nhwc_cython(self, x: np.ndarray) -> np.ndarray:        
        y = self.y[:x.shape[0] , :]        
        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, self.id * PYDTNN_OPS_EVENTS + PYDTNN_OPS_EVENT_enum.FORWARD_ADP_AVG_POOL)
        adaptive_avg_pooling_fwd_nhwc_cython(x, y)
        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, PYDTNN_EVENT_FINISHED)        
        return y

    def _forward_nchw_cython(self, x: np.ndarray) -> np.ndarray:
        y = self.y[:x.shape[0], :]
        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, self.id * PYDTNN_OPS_EVENTS + PYDTNN_OPS_EVENT_enum.FORWARD_ADP_AVG_POOL)        
        adaptive_avg_pooling_fwd_nchw_cython(x, y)
        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, PYDTNN_EVENT_FINISHED)
        
        return y

    def _backward_nhwc_cython(self, dy: np.ndarray) -> np.ndarray:
        dx = self.dx[:dy.shape[0],:]
        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, self.id * PYDTNN_OPS_EVENTS + PYDTNN_OPS_EVENT_enum.BACKWARD_ADP_AVG_POOL)
        adaptive_avg_pooling_bwd_nhwc_cython(dy, dx)
        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, PYDTNN_EVENT_FINISHED)
        return dx
    # --- END _backward_nhwc_cython --- #
        
    def _backward_nchw_cython(self, dy: np.ndarray) -> np.ndarray:
        dx = self.dx[:dy.shape[0], :]
        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, self.id * PYDTNN_OPS_EVENTS + PYDTNN_OPS_EVENT_enum.BACKWARD_ADP_AVG_POOL)
        adaptive_avg_pooling_bwd_nchw_cython(dy, dx)
        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, PYDTNN_EVENT_FINISHED)
        return dx
    # --- END _backward_nchw_cython --- #    
# --- END AdaptiveAveragePool2DCPU --- #
