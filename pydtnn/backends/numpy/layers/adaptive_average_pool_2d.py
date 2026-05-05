import logging
from typing import TYPE_CHECKING

from pydtnn.backends.numpy.layers.abstract.pool_2d_layer import AbstractPool2DLayerNumpy
from pydtnn.backends.numpy.layers.layer import LayerNumpy
from pydtnn.layers.adaptive_average_pool_2d import AdaptiveAveragePool2D
from pydtnn.libs import numpy as np
from pydtnn.tracers.events import PYDTNN_EVENT_FINISHED, PYDTNN_OPS_EVENT, PYDTNN_OPS_EVENTS, PYDTNN_OPS_EVENT_enum
from pydtnn.utils.constants import ArrayShape
from pydtnn.utils.tensor import TensorFormat

__all__ = (
    "AdaptiveAveragePool2DNumpy",
)


logger = logging.getLogger(__name__)


# Imports for the method from AbstractPool2DLayerNumpy

# Imports for the methods from AveragePool2DNumpy
if TYPE_CHECKING:
    import numpy as np


class AdaptiveAveragePool2DNumpy(AdaptiveAveragePool2D[np.ndarray], AbstractPool2DLayerNumpy):
    # The backend is almost the same as a AveragePool2D layer.

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # The following atributte will be initalized in "initalize"
        self.y: np.ndarray = None  # type: ignore

    # Method from AbstractPool2DLayerNumpy
    def _model_init(self, prev_shape: ArrayShape, x: np.ndarray | None = None):
        # The objective is following lines is to override the
        # AbstractPool2DLayer's initialize method, that is avoiding call to
        # "super" since in that case AbstractPool2DLayer will be called
        # eventually.
        super()._model_init(prev_shape, x)

        match self.model.tensor_format:
            case TensorFormat.NCHW:
                self._forward = self._forward_nchw
                self._backward = self._backward_nchw
            case TensorFormat.NHWC:
                self._forward = self._forward_nhwc
                self._backward = self._backward_nhwc
            case _:
                raise NotImplementedError(f"AdaptiveAveragePool2DNumpy is not implemented for {self.model.tensor_format} format.")

        y_shape = self.model.encode_shape((self.model.batch_size, self.co, self.ho, self.wo))
        # NOTE: This attribute only stores data, its value before the operation doesn't matter; it's initalized due avoid warnings in "LayerAndActivationBase.export".
        self.y = np.zeros(y_shape, dtype=self.model.dtype)
        self.memory_used += self.y.nbytes

        if not self.model.evaluate_only:
            dx_shape = self.model.encode_shape((self.model.batch_size, self.ci, self.hi, self.wi))
            self.dx = np.zeros(dx_shape, dtype=self.model.dtype)
            self.memory_used += self.dx.nbytes

        if self.pooling_not_needed:
            self._forward = (lambda x: x)
        # else: Nothing special.

    def forward(self, x: np.ndarray) -> np.ndarray:
        return self._forward(x)

    def backward(self, dy: np.ndarray) -> np.ndarray:
        return self._backward(dy)

    def _fwd_nchw(self, x: np.ndarray, y: np.ndarray) -> None:
        for nn in range(x.shape[0]):
            for cc in range(self.ci):
                for hi in range(self.ho):
                    h_start = AdaptiveAveragePool2D._index_first_element(hi, self.hi, self.ho)
                    h_end = AdaptiveAveragePool2D._index_last_element(hi, self.hi, self.ho)
                    elements_h = h_end - h_start

                    for wi in range(self.wo):
                        w_start = AdaptiveAveragePool2D._index_first_element(wi, self.wi, self.wo)
                        w_end = AdaptiveAveragePool2D._index_last_element(wi, self.wi, self.wo)
                        elements = elements_h * (w_end - w_start)

                        add = 0
                        for i in range(h_start, h_end):
                            for j in range(w_start, w_end):
                                add += x[nn, cc, i, j]
                        y[nn, cc, hi, wi] = add / elements

    def _fwd_nhwc(self, x: np.ndarray, y: np.ndarray) -> None:
        for nn in range(x.shape[0]):
            for cc in range(self.ci):
                for hi in range(self.ho):
                    h_start = AdaptiveAveragePool2D._index_first_element(hi, self.hi, self.ho)
                    h_end = AdaptiveAveragePool2D._index_last_element(hi, self.hi, self.ho)
                    elements_h = h_end - h_start

                    for wi in range(self.wo):
                        w_start = AdaptiveAveragePool2D._index_first_element(wi, self.wi, self.wo)
                        w_end = AdaptiveAveragePool2D._index_last_element(wi, self.wi, self.wo)
                        elements = elements_h * (w_end - w_start)

                        add = 0
                        for i in range(h_start, h_end):
                            for j in range(w_start, w_end):
                                add += x[nn, i, j, cc]
                        y[nn, hi, wi, cc] = add / elements

    def _bwd_nchw(self, dx: np.ndarray, dy: np.ndarray) -> None:
        for nn in range(dy.shape[0]):
            for cc in range(self.ci):
                for ho in range(self.ho):
                    h_start = AdaptiveAveragePool2D._index_first_element(ho, self.ho, self.hi)
                    h_end = AdaptiveAveragePool2D._index_last_element(ho, self.ho, self.hi)
                    elements_h = h_end - h_start
                    for wo in range(self.wo):
                        w_start = AdaptiveAveragePool2D._index_first_element(wo, self.wo, self.wi)
                        w_end = AdaptiveAveragePool2D._index_last_element(wo, self.wo, self.wi)
                        elements = elements_h * (w_end - w_start)

                        delta = dy[nn, cc, ho, wo] / elements
                        for i in range(h_start, h_end):
                            for j in range(w_start, w_end):
                                dx[nn, cc, i, j] += delta

    def _bwd_nhwc(self, dx: np.ndarray, dy: np.ndarray) -> None:
        for nn in range(dy.shape[0]):
            for cc in range(self.ci):
                for ho in range(self.ho):
                    h_start = AdaptiveAveragePool2D._index_first_element(ho, self.ho, self.hi)
                    h_end = AdaptiveAveragePool2D._index_last_element(ho, self.ho, self.hi)
                    elements_h = h_end - h_start
                    for wo in range(self.wo):
                        w_start = AdaptiveAveragePool2D._index_first_element(wo, self.wo, self.wi)
                        w_end = AdaptiveAveragePool2D._index_last_element(wo, self.wo, self.wi)
                        elements = elements_h * (w_end - w_start)

                        delta = dy[nn, ho, wo, cc] / elements
                        for i in range(h_start, h_end):
                            for j in range(w_start, w_end):
                                dx[nn, i, j, cc] += delta

    def _forward_nhwc(self, x: np.ndarray) -> np.ndarray:
        y: np.ndarray = np.ascontiguousarray(self.y[:x.shape[0], :], dtype=self.model.dtype)
        self.mask = np.ascontiguousarray(self._mask[:x.shape[0], :], dtype=self.model.dtype)

        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, self.id * PYDTNN_OPS_EVENTS + PYDTNN_OPS_EVENT_enum.FORWARD_ADP_AVG_POOL)
        self._fwd_nhwc(x, y)
        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, PYDTNN_EVENT_FINISHED)
        return np.asarray(y, dtype=self.model.dtype, order="C")

    def _forward_nchw(self, x: np.ndarray) -> np.ndarray:
        y: np.ndarray = np.ascontiguousarray(self.y[:x.shape[0], :], dtype=self.model.dtype)
        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, self.id * PYDTNN_OPS_EVENTS + PYDTNN_OPS_EVENT_enum.FORWARD_ADP_AVG_POOL)
        self._fwd_nchw(x, y)
        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, PYDTNN_EVENT_FINISHED)
        return np.asarray(y, dtype=self.model.dtype, order="C")

    def _backward_nhwc(self, dy: np.ndarray) -> np.ndarray:
        dx: np.ndarray = np.ascontiguousarray(self.dx[:dy.shape[0], :], dtype=self.model.dtype)
        dx.fill(0)
        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, self.id * PYDTNN_OPS_EVENTS + PYDTNN_OPS_EVENT_enum.BACKWARD_ADP_AVG_POOL)
        self._bwd_nhwc(dx, dy)
        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, PYDTNN_EVENT_FINISHED)
        return np.asarray(dx, dtype=self.model.dtype, order="C")

    def _backward_nchw(self, dy: np.ndarray) -> np.ndarray:
        dx: np.ndarray = np.ascontiguousarray(self.dx[:dy.shape[0], :], dtype=self.model.dtype)
        dx.fill(0)
        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, self.id * PYDTNN_OPS_EVENTS + PYDTNN_OPS_EVENT_enum.BACKWARD_ADP_AVG_POOL)
        self._bwd_nchw(dx, dy)
        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, PYDTNN_EVENT_FINISHED)
        return np.asarray(dx, dtype=self.model.dtype, order="C")
