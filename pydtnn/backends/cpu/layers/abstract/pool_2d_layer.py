from pydtnn.backends.cpu.layers.layer import LayerCPU
from pydtnn.layers.abstract.pool_2d_layer import AbstractPool2DLayer
from pydtnn.utils.performance_models import im2col_time, col2im_time
from pydtnn.utils.tensor import TensorFormat
from pydtnn.utils.constants import ArrayShape

import numpy as np


class AbstractPool2DLayerCPU(AbstractPool2DLayer[np.ndarray], LayerCPU):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def initialize(self, prev_shape: ArrayShape, x: np.ndarray | None = None):
        super().initialize(prev_shape, x)

        match self.model.tensor_format:
            case TensorFormat.NCHW:
                self.forward = self._forward_nchw_cython
                self.backward = self._backward_nchw_cython
            case TensorFormat.NHWC:
                self.forward = self._forward_nhwc_cython
                self.backward = self._backward_nhwc_cython
            case _:
                raise TypeError(f"Function: \'AbstractPool2DLayerCPU\'. Error:\n\tFormat: \'{self.model.tensor_format}\' not supported.")

        # I2C-based implementations have been temporarily discarded
        # setattr(self, "forward", self._forward_nchw_i2c)
        # setattr(self, "backward", self._backward_nchw_i2c)
        # setattr(self, "forward", self._forward_nhwc_i2c)
        # setattr(self, "backward", self._backward_nhwc_i2c)

        # The following variable is only for NCHW implementation (not for i2c implementation)
        y_shape = self.model.encode_shape((self.model.batch_size, self.co, self.ho, self.wo))
        # NOTE: This attribute only stores data, its value before the operation doesn't matter; it's initalized due avoid warnings in "LayerAndActivationBase.export".
        self.y = np.zeros(y_shape, dtype=self.model.dtype, order="C")

        self.fwd_time = \
            im2col_time(m=(self.kh * self.kw), n=(self.model.batch_size * self.ho * self.wo * self.ci),
                        cpu_speed=self.model.cpu_speed, memory_bw=self.model.memory_bw,
                        dtype=self.model.dtype)
        self.bwd_time = \
            col2im_time(m=(self.kh * self.kw), n=(self.model.batch_size * self.ho * self.wo * self.ci),
                        cpu_speed=self.model.cpu_speed, memory_bw=self.model.memory_bw,
                        dtype=self.model.dtype)

    def forward(self, x: np.ndarray) -> np.ndarray:
        msg = """This is a fake forward function. It will be masked on initialization by _forward_i2c or _forward_cg"""
        raise NotImplementedError(f"Class \'AbstractPool2DLayerCPU\'. Error: {msg}")

    def backward(self, dy: np.ndarray) -> np.ndarray:
        msg = """This is a fake backward function. It will be masked on initialization by _backward_i2c or _backward_cg"""
        raise NotImplementedError(f"Class \'AbstractPool2DLayerCPU\'. Error: {msg}")
    # ---

    def _forward_nchw_cython(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError()

    def _backward_nchw_cython(self, dy: np.ndarray) -> np.ndarray:
        raise NotImplementedError()

    def _forward_nhwc_cython(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError()

    def _backward_nhwc_cython(self, dy: np.ndarray) -> np.ndarray:
        raise NotImplementedError()
