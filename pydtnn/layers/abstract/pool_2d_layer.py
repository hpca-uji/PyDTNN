import math
from pydtnn.utils.constants import Array
from pydtnn.layers.layer import Layer, LayerError
import logging
logger = logging.getLogger(__name__)


class AbstractPool2DLayer[T: Array](Layer[T]):

    def __init__(self, pool_shape: tuple[int, int] | int = (2, 2), padding: tuple[int, int] | int = 0,
                 stride: tuple[int, int] | int = 1, dilation: tuple[int, int] | int = 1):
        super().__init__()
        self.pool_shape = (pool_shape, pool_shape) if isinstance(pool_shape, int) else pool_shape
        self.padding = padding
        self.stride = stride
        self.dilation = dilation
        self.hpadding, self.wpadding = (padding, padding) if isinstance(padding, int) else padding
        self.hstride, self.wstride = (stride, stride) if isinstance(stride, int) else stride
        self.hdilation, self.wdilation = (dilation, dilation) if isinstance(dilation, int) else dilation
        self.ci = self.hi = self.wi = self.kh = self.kw = self.ho = self.wo = self.co = self.n = 0

    def _model_init(self, prev_shape, x: T | None):
        super()._model_init(prev_shape, x)
        self.ci, self.hi, self.wi = self.model.decode_shape(prev_shape)
        if self.pool_shape[0] == 0:
            self.pool_shape = (self.hi, self.pool_shape[1])
        if self.pool_shape[1] == 0:
            self.pool_shape = (self.pool_shape[0], self.wi)
        self.kh, self.kw = self.pool_shape
        self.co = self.ci
        self.ho = (self.hi + 2 * self.hpadding - self.hdilation * (self.kh - 1) - 1) // self.hstride + 1
        self.wo = (self.wi + 2 * self.wpadding - self.wdilation * (self.kw - 1) - 1) // self.wstride + 1
        if not (self.ho > 0 and self.wo > 0):
            raise LayerError(f"Output dimensions must be greater than 0. ho: {self.ho}, wo: {self.wo}.")
        self.shape = self.model.encode_shape((self.co, self.ho, self.wo))
        self.n = math.prod(self.shape)

    def _show_props(self) -> dict:
        props = super()._show_props()

        props["pool"] = self.pool_shape
        props["padding"] = (self.hpadding, self.wpadding)
        props["stride"] = (self.hstride, self.wstride)
        props["dilation"] = (self.hdilation, self.wdilation)

        return props
