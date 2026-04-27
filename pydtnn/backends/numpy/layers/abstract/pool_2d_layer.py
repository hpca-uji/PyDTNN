import logging
import math
from typing import TYPE_CHECKING

from pydtnn.backends.numpy.layers.layer import LayerNumpy
from pydtnn.layers.abstract.pool_2d_layer import AbstractPool2DLayer
from pydtnn.libs import numpy as np
from pydtnn.utils.constants import ArrayShape
from pydtnn.utils.performance_models import col2im_time, im2col_time
from pydtnn.utils.tensor import TensorFormat

logger = logging.getLogger(__name__)


if TYPE_CHECKING:
    import numpy as np


class AbstractPool2DLayerNumpy(AbstractPool2DLayer[np.ndarray], LayerNumpy):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _model_init(self, prev_shape: ArrayShape, x: np.ndarray | None = None):
        super()._model_init(prev_shape, x)

        match self.model.tensor_format:
            case TensorFormat.NCHW:
                self.forward = self._forward_nchw
                self.backward = self._backward_nchw
            case TensorFormat.NHWC:
                self.forward = self._forward_nhwc
                self.backward = self._backward_nhwc
            case _:
                raise TypeError(f"Function: \'AbstractPool2DLayerNumpy\'. Error:\n\tFormat: \'{self.model.tensor_format}\' not supported.")

        # I2C-based implementations have been temporarily discarded
        # setattr(self, "forward", self._forward_nchw_i2c)
        # setattr(self, "backward", self._backward_nchw_i2c)
        # setattr(self, "forward", self._forward_nhwc_i2c)
        # setattr(self, "backward", self._backward_nhwc_i2c)

        # The following variable is only for NCHW implementation (not for i2c implementation)
        # y_shape = self.model.encode_shape((self.model.batch_size, self.co, self.ho, self.wo))
        # NOTE: This attribute only stores data, its value before the operation doesn't matter; it's initalized due avoid warnings in "LayerAndActivationBase.export".
        # self.y = np.zeros(y_shape, dtype=self.model.dtype)
        # self.real_memory_size += self.y.nbytes
        self.y_size = self.model.batch_size * self.co * self.ho * self.wo

        if not self.model.evaluate_only:
            # dx_shape = self.model.encode_shape((self.model.batch_size, self.ci, self.hi, self.wi))
            self.dx_size = self.model.batch_size * self.ci * self.hi * self.wi
            # self.dx = np.zeros(dx_shape, dtype=self.model.dtype)
            # self.real_memory_size += self.dx.nbytes
        else:
            self.dx_size = 0

        self.y_dx = np.zeros(shape=(max(self.y_size, self.dx_size), ), dtype=self.model.dtype)
        # NOTE: self.y_dx stores both y and dx values.
        self.memory_used += self.y_dx.nbytes

        self.fwd_time = \
            im2col_time(m=(self.kh * self.kw), n=(self.model.batch_size * self.ho * self.wo * self.ci),
                        cpu_speed=self.model.cpu_speed, memory_bw=self.model.memory_bw,
                        dtype=self.model.dtype)  # type: ignore (it's fine)
        self.bwd_time = \
            col2im_time(m=(self.kh * self.kw), n=(self.model.batch_size * self.ho * self.wo * self.ci),
                        cpu_speed=self.model.cpu_speed, memory_bw=self.model.memory_bw,
                        dtype=self.model.dtype)  # type: ignore (it's fine)
    # ----

    def get_y(self, batch_size: int) -> np.ndarray:
        y_shape = self.model.encode_shape((batch_size, self.co, self.ho, self.wo))
        y_size = math.prod(y_shape)
        y = self.y_dx[:y_size]
        return np.ascontiguousarray(y.reshape(y_shape), dtype=self.model.dtype)

    def get_dx(self, batch_size: int) -> np.ndarray:
        dx_shape = self.model.encode_shape((batch_size, self.ci, self.hi, self.wi))
        dx_size = math.prod(dx_shape)
        dx = self.y_dx[:dx_size]
        return np.ascontiguousarray(dx.reshape(dx_shape), dtype=self.model.dtype)

    def forward(self, x: np.ndarray) -> np.ndarray:
        msg = """This is a fake forward function. It will be masked on initialization by _forward_i2c or _forward_cg"""
        raise NotImplementedError(f"Class \'AbstractPool2DLayerNumpy\'. Error: {msg}")

    def backward(self, dy: np.ndarray) -> np.ndarray:
        msg = """This is a fake backward function. It will be masked on initialization by _backward_i2c or _backward_cg"""
        raise NotImplementedError(f"Class \'AbstractPool2DLayerNumpy\'. Error: {msg}")
    # ---

    def _forward_nchw(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError(f"This is a fake method. {self} must implement _forward_nchw.")

    def _backward_nchw(self, dy: np.ndarray) -> np.ndarray:
        raise NotImplementedError(f"This is a fake method. {self} must implement _backward_nchw.")

    def _forward_nhwc(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError(f"This is a fake method. {self} must implement _forward_nhwc.")

    def _backward_nhwc(self, dy: np.ndarray) -> np.ndarray:
        raise NotImplementedError(f"This is a fake method. {self} must implement _backward_nhwc.")
