from pydtnn.libs import numpy as np
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    import numpy as np

from pydtnn.backends.cpu.layers.abstract.pool_2d_layer import AbstractPool2DLayerCPU
from pydtnn.layers.max_pool_2d import MaxPool2D
from pydtnn.tracers.events import PYDTNN_OPS_EVENT, PYDTNN_OPS_EVENTS, PYDTNN_EVENT_FINISHED, PYDTNN_OPS_EVENT_enum
from pydtnn.utils.constants import ArrayShape


class MaxPool2DCPU(MaxPool2D[np.ndarray], AbstractPool2DLayerCPU):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # The following attribute will be intialized later.
        self.idx_max: np.ndarray = None  # type: ignore
        self.y: np.ndarray  # NOTE: Defined and initalized in AbstractPool2DLayerCPU's init and initialize, respectively

    def initialize(self, prev_shape: ArrayShape, x: np.ndarray | None = None):
        super().initialize(prev_shape, x)
        self.minval = int(np.iinfo(self.model.dtype).min) if np.issubdtype(self.model.dtype, np.integer) else float(np.finfo(self.model.dtype).min)
        idx_max_shape = self.model.encode_shape((self.model.batch_size, self.co, self.ho, self.wo))

        # NOTE: This attribute only stores data, its value before the operation doesn't matter; it's initalized due avoid warnings in "LayerAndActivationBase.export".
        self._idx_max: np.ndarray = np.zeros(idx_max_shape, dtype=np.int32)
        self.real_memory_size += self._idx_max.nbytes
    # ---

    def _forward_nhwc(self, x: np.ndarray) -> np.ndarray:

        # y:np.ndarray = self.y[:x.shape[0], :]
        y = self.get_y(x.shape[0])
        self.idx_max: np.ndarray = self._idx_max[:x.shape[0], :]

        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, self.id * PYDTNN_OPS_EVENTS + PYDTNN_OPS_EVENT_enum.FORWARD_IM2COL)
        for nn in range(x.shape[0]):
            for xx in range(self.ho):
                for yy in range(self.wo):
                    for cc in range(self.ci):
                        maxval = self.minval
                        idx_maxval = 0
                        for ii in range(self.kh):
                            x_x = self.vstride * xx + self.vdilation * ii - self.vpadding
                            if 0 <= x_x < self.hi:
                                for jj in range(self.kw):
                                    x_y = self.hstride * yy + self.hdilation * jj - self.hpadding
                                    if 0 <= x_y < self.wi:
                                        val = x[nn, x_x, x_y, cc]
                                        if val > maxval:
                                            maxval = val
                                            idx_maxval = ii * self.kw + jj
                        y[nn, xx, yy, cc] = maxval
                        self.idx_max[nn, xx, yy, cc] = idx_maxval
        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, PYDTNN_EVENT_FINISHED)
        return y

    def _forward_nchw(self, x: np.ndarray) -> np.ndarray:
        # y:np.ndarray = self.y[:x.shape[0], :]
        y = self.get_y(x.shape[0])
        self.idx_max = self._idx_max[:x.shape[0], :]

        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, self.id * PYDTNN_OPS_EVENTS + PYDTNN_OPS_EVENT_enum.FORWARD_IM2COL)
        for nn in range(x.shape[0]):
            for cc in range(self.ci):
                for xx in range(self.ho):
                    for yy in range(self.wo):
                        maxval = self.minval
                        idx_maxval = 0
                        for ii in range(self.kh):
                            x_x = self.vstride * xx + self.vdilation * ii - self.vpadding
                            if 0 <= x_x < self.hi:
                                for jj in range(self.kw):
                                    x_y = self.hstride * yy + self.hdilation * jj - self.hpadding
                                    if 0 <= x_y < self.wi:
                                        val = x[nn, cc, x_x, x_y]
                                        if val > maxval:
                                            maxval = val
                                            idx_maxval = ii * self.kw + jj
                        y[nn, cc, xx, yy] = maxval
                        self.idx_max[nn, cc, xx, yy] = idx_maxval
        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, PYDTNN_EVENT_FINISHED)
        return np.asarray(y, dtype=self.model.dtype)

    def _backward_nhwc(self, dy: np.ndarray) -> np.ndarray:
        # dx:np.ndarray = self.dx[ :dy.shape[0], :]
        dx = self.get_dx(dy.shape[0])
        dx.fill(0)
        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, self.id * PYDTNN_OPS_EVENTS + PYDTNN_OPS_EVENT_enum.COMP_DX_COL2IM)
        for nn in range(dy.shape[0]):
            for xx in range(self.ho):
                for yy in range(self.wo):
                    for cc in range(self.ci):
                        idx_maxval = self.idx_max[nn, xx, yy, cc]
                        ii = idx_maxval // self.kh
                        jj = idx_maxval % self.kw
                        x_x = self.vstride * xx + self.vdilation * ii - self.vpadding
                        x_y = self.hstride * yy + self.hdilation * jj - self.hpadding
                        if 0 <= x_x < self.ho and 0 <= x_y < self.wo:
                            dx[nn, x_x, x_y, cc] += dy[nn, xx, yy, cc]
        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, PYDTNN_EVENT_FINISHED)
        return dx

    def _backward_nchw(self, dy: np.ndarray) -> np.ndarray:

        # dx:np.ndarray = self.dx[ :dy.shape[0], :]
        dx = self.get_dx(dy.shape[0])
        dx.fill(0)
        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, self.id * PYDTNN_OPS_EVENTS + PYDTNN_OPS_EVENT_enum.COMP_DX_COL2IM)
        for nn in range(dy.shape[0]):
            for cc in range(self.ci):
                for xx in range(self.ho):
                    for yy in range(self.wo):
                        idx_maxval = self.idx_max[nn, cc, xx, yy]
                        ii = idx_maxval // self.kh
                        jj = idx_maxval % self.kw
                        x_x = self.vstride * xx + self.vdilation * ii - self.vpadding
                        x_y = self.hstride * yy + self.hdilation * jj - self.hpadding
                        if 0 <= x_x < self.ho and 0 <= x_y < self.wo:
                            dx[nn, cc, x_x, x_y] += dy[nn, cc, xx, yy]

        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, PYDTNN_EVENT_FINISHED)
        return np.asarray(dx, dtype=self.model.dtype)

    ##########
    ## TEST ##
    ##########
    
    def max_pool(self, x: np.ndarray, y: np.ndarray, idx_maxval: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        x = np.pad(x, ((0,0), (0,0), (self.vpadding, self.vpadding), (self.hpadding, self.hpadding)), mode="constant")
        for kh in range(self.kh):
            for kw in range(self.kw):
                h_start = kh * self.vdilation
                w_start = kw * self.hdilation
                h_end = h_start + self.vstride * self.ho
                w_end = w_start + self.hstride * self.wo
                
                _x = x[:, :, h_start:h_end:self.vstride, w_start:w_end:self.hstride]
                max_val: np.ndarray = np.max(_x, axis=(2, 3))
                _idx_maxval: np.ndarray = np.argmax(np.argmax(_x, axis=3), axis=2)

                #y[:, :, h_start:h_end:self.vstride, w_start:w_end:self.hstride] = max_val[:, :]
                #idx_maxval[:, :, h_start:h_end:self.vstride, w_start:w_end:self.hstride] = _idx_maxval[:, :]
                #breakpoint()
                for i in range(h_start, h_end // self.vstride):
                    for j in range(w_start, w_end // self.hstride):
                        y[:, :, i, j] = max_val[:, :]
                        idx_maxval[:, :, i, j] = _idx_maxval[:, :]
                #breakpoint()
        return (y, idx_maxval)
