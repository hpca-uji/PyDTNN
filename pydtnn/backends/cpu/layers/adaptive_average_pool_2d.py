from pydtnn.backends.cpu.utils.adaptive_avg_pooling_nchw_cython import adaptive_avg_pooling_bwd_nchw_cython, adaptive_avg_pooling_fwd_nchw_cython
from pydtnn.backends.cpu.utils.adaptive_avg_pooling_nhwc_cython import adaptive_avg_pooling_bwd_nhwc_cython, adaptive_avg_pooling_fwd_nhwc_cython
from pydtnn.layers.adaptive_average_pool_2d import AdaptiveAveragePool2D
from pydtnn.backends.cpu.layers.layer import LayerCPU

# Imports for the method from AbstractPool2DLayerCPU
from pydtnn.utils.tensor import TensorFormat

# Imports for the methods from AveragePool2DCPU
from pydtnn.tracers.events import PYDTNN_OPS_EVENT, PYDTNN_OPS_EVENTS, PYDTNN_EVENT_FINISHED, PYDTNN_OPS_EVENT_enum
import numpy as np


class AdaptiveAveragePool2DCPU(AdaptiveAveragePool2D[np.ndarray], LayerCPU):
    # The backend is almost the same as a AveragePool2D layer.

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # The following atributte will be initalized in "initalize"
        self.y: np.ndarray = None  # type: ignore
    # -- END __init__ -- #

    # Method from AbstractPool2DLayerCPU
    def initialize(self, prev_shape: tuple[int, int], x: np.ndarray | None = None):
        # The objective is following lines is to override the AbstractPool2DLayer's initialize method, that is avoiding call to "super" since in that case AbstractPool2DLayer will be called eventually.
        super().initialize(prev_shape, x)

        match self.model.tensor_format:
            case TensorFormat.NCHW:
                self._forward = self._forward_nchw_cython
                self._backward = self._backward_nchw_cython
            case TensorFormat.NHWC:
                self._forward = self._forward_nhwc_cython
                self._backward = self._backward_nhwc_cython
            case _:
                raise NotImplementedError(f"\"AdaptiveAveragePool2DCPU\" is not implemented for \"{self.model.tensor_format}\" format.")

        y_shape = self.model.encode_shape((self.model.batch_size, self.co, self.ho, self.wo))
        # NOTE: This attribute only stores data, its value before the operation doesn't matter; it's initalized due avoid warnings in "LayerAndActivationBase.export".
        self.y = np.zeros(y_shape, dtype=self.model.dtype, order="C")

        if self.pooling_not_needed:
            self._forward = (lambda x: x)
        # else: Nothing special.

    # -- END initialize -- #

    def forward(self, x: np.ndarray) -> np.ndarray:
        return self._forward(x)

    def backward(self, dy: np.ndarray) -> np.ndarray:
        return self._backward(dy)

    def _forward_nhwc_cython(self, x: np.ndarray) -> np.ndarray:
        y = self.y[:x.shape[0], :]
        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, self.id * PYDTNN_OPS_EVENTS + PYDTNN_OPS_EVENT_enum.FORWARD_ADP_AVG_POOL)
        adaptive_avg_pooling_fwd_nhwc_cython(x, y)
        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, PYDTNN_EVENT_FINISHED)
        return np.asarray(y, dtype=self.model.dtype, order='C', copy=None)

    def _forward_nchw_cython(self, x: np.ndarray) -> np.ndarray:
        y = self.y[:x.shape[0], :]
        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, self.id * PYDTNN_OPS_EVENTS + PYDTNN_OPS_EVENT_enum.FORWARD_ADP_AVG_POOL)
        adaptive_avg_pooling_fwd_nchw_cython(x, y)
        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, PYDTNN_EVENT_FINISHED)
        return np.asarray(y, dtype=self.model.dtype, order='C', copy=None)

    def _backward_nhwc_cython(self, dy: np.ndarray) -> np.ndarray:
        dx = np.zeros((dy.shape[0], self.hi, self.wi, self.ci), dtype=self.model.dtype, order="C")
        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, self.id * PYDTNN_OPS_EVENTS + PYDTNN_OPS_EVENT_enum.BACKWARD_ADP_AVG_POOL)
        adaptive_avg_pooling_bwd_nhwc_cython(dy, dx)
        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, PYDTNN_EVENT_FINISHED)
        return np.asarray(dx, dtype=self.model.dtype, order='C', copy=None)

    def _backward_nchw_cython(self, dy: np.ndarray) -> np.ndarray:
        dx = np.zeros((dy.shape[0], self.ci, self.hi, self.wi), dtype=self.model.dtype, order="C")
        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, self.id * PYDTNN_OPS_EVENTS + PYDTNN_OPS_EVENT_enum.BACKWARD_ADP_AVG_POOL)
        adaptive_avg_pooling_bwd_nchw_cython(dy, dx)
        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, PYDTNN_EVENT_FINISHED)
        return np.asarray(dx, dtype=self.model.dtype, order='C', copy=None)
