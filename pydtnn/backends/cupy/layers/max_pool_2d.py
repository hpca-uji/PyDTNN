import cupy as np

from pydtnn.backends.cupy.layers.abstract.pool_2d_layer import AbstractPool2DLayerCUPY
from pydtnn.layers.max_pool_2d import MaxPool2D
from pydtnn.tracers.events import PYDTNN_OPS_EVENT, PYDTNN_OPS_EVENTS, PYDTNN_EVENT_FINISHED, PYDTNN_OPS_EVENT_enum
from pydtnn.utils.constants import ArrayShape


class MaxPool2DCUPY(MaxPool2D[np.ndarray], AbstractPool2DLayerCUPY):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # The following attribute will be intialized later.
        self.idx_max: np.ndarray = None  # type: ignore
        self.y: np.ndarray

    def initialize(self, prev_shape: ArrayShape, x: np.ndarray | None = None):
        super().initialize(prev_shape, x)
        self.minval = np.iinfo(self.model.dtype).min if np.issubdtype(self.model.dtype, np.integer) else np.finfo(self.model.dtype).min
        idx_max_shape = self.model.encode_shape((self.model.batch_size, self.co, self.ho, self.wo))
        # NOTE: This attribute only stores data, its value before the operation doesn't matter; it's initalized due avoid warnings in "LayerAndActivationBase.export".
        self._idx_max = np.zeros(idx_max_shape, dtype=np.int32)

    def _forward_nhwc_cython(self, x: np.ndarray) -> np.ndarray:

        y = self.y[:x.shape[0], :]
        self.idx_max = self._idx_max[:x.shape[0], :]
        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, self.id * PYDTNN_OPS_EVENTS + PYDTNN_OPS_EVENT_enum.FORWARD_IM2COL)
        max_pool_2d_fwd_nhwc_cython(x, y, self.idx_max,
                                    self.kh, self.kw, self.ho, self.wo,
                                    self.vpadding, self.hpadding,
                                    self.vstride, self.hstride,
                                    self.vdilation, self.hdilation,
                                    self.minval)
        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, PYDTNN_EVENT_FINISHED)
        return y

    def _forward_nchw_cython(self, x: np.ndarray) -> np.ndarray:
        y = self.y[:x.shape[0], :]
        self.idx_max = self._idx_max[:x.shape[0], :]

        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, self.id * PYDTNN_OPS_EVENTS + PYDTNN_OPS_EVENT_enum.FORWARD_IM2COL)
        max_pool_2d_fwd_nchw_cython(x, y, self.idx_max,
                                    self.kh, self.kw, self.ho, self.wo,
                                    self.vpadding, self.hpadding,
                                    self.vstride, self.hstride,
                                    self.vdilation, self.hdilation,
                                    self.minval)
        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, PYDTNN_EVENT_FINISHED)
        return np.asarray(y, dtype=self.model.dtype)

    def _backward_nhwc_cython(self, dy: np.ndarray) -> np.ndarray:
        dx = np.zeros((dy.shape[0], self.hi, self.wi, self.ci), dtype=self.model.dtype)
        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, self.id * PYDTNN_OPS_EVENTS + PYDTNN_OPS_EVENT_enum.COMP_DX_COL2IM)
        max_pool_2d_bwd_nhwc_cython(dy, self.idx_max, dx,
                                    dy.shape[0], self.hi, self.wi, self.ci,
                                    self.kh, self.kw, self.ho, self.wo,
                                    self.vpadding, self.hpadding,
                                    self.vstride, self.hstride, self.vdilation, self.hdilation)
        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, PYDTNN_EVENT_FINISHED)
        return dx

    def _backward_nchw_cython(self, dy: np.ndarray) -> np.ndarray:

        dx = np.zeros((dy.shape[0], self.ci, self.hi, self.wi), dtype=self.model.dtype)
        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, self.id * PYDTNN_OPS_EVENTS + PYDTNN_OPS_EVENT_enum.COMP_DX_COL2IM)
        max_pool_2d_bwd_nchw_cython(dy, self.idx_max, dx,
                                    dy.shape[0], self.hi, self.wi, self.ci,
                                    self.kh, self.kw, self.ho, self.wo,
                                    self.vpadding, self.hpadding,
                                    self.vstride, self.hstride, self.vdilation, self.hdilation)
        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, PYDTNN_EVENT_FINISHED)
        return np.asarray(dx, dtype=self.model.dtype)

    def fwd_kernel(self) -> str:
        code = \
            """



"""
        code.format()
        return code
