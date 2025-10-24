from pydtnn.backends.cpu.layers.layer_cpu import LayerCPU
from pydtnn.layers.abstract_pool_2d_layer import AbstractPool2DLayer
from pydtnn.performance_models import im2col_time, col2im_time
from pydtnn.utils.tensor import TensorFormat
from numpy import ndarray, empty
from pydtnn.utils.types import ArrayShape


class AbstractPool2DLayerCPU(LayerCPU, AbstractPool2DLayer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def initialize(self, prev_shape: ArrayShape, x: ndarray | None = None):
        super().initialize(prev_shape, x)

        match self.model.tensor_format:
            case TensorFormat.NCHW:
                self.forward = self._forward_nchw_cython
                self.backward = self._backward_nchw_cython

                # The following variable is only for NCHW implementation (not for i2c implementation)
                self.y = empty((self.model.batch_size, self.co, self.ho, self.wo), dtype=self.model.dtype, order="C")

                # I2C-based implementations have been temporarily discarded
                # setattr(self, "forward", self._forward_nchw_i2c)
                # setattr(self, "backward", self._backward_nchw_i2c)
            case TensorFormat.NHWC:
                self.forward = self._forward_nhwc_cython
                self.backward = self._backward_nhwc_cython

                # The following variable is only for NHWC implementation (not for i2c implementation)
                self.y = empty((self.model.batch_size, self.ho, self.wo, self.co), dtype=self.model.dtype)

                # I2C-based implementations have been temporarily discarded
                # setattr(self, "forward", self._forward_nhwc_i2c)
                # setattr(self, "backward", self._backward_nhwc_i2c)
            case _:
                raise TypeError(f"Function: \'AbstractPool2DLayerCPU\'. Error:\n\tFormat: \'{self.model.tensor_format}\' not supported.")

        self.fwd_time = \
            im2col_time(m=(self.kh * self.kw), n=(self.model.batch_size * self.ho * self.wo * self.ci),
                        cpu_speed=self.model.cpu_speed, memory_bw=self.model.memory_bw,
                        dtype=self.model.dtype)
        self.bwd_time = \
            col2im_time(m=(self.kh * self.kw), n=(self.model.batch_size * self.ho * self.wo * self.ci),
                        cpu_speed=self.model.cpu_speed, memory_bw=self.model.memory_bw,
                        dtype=self.model.dtype)

    def forward(self, x: ndarray) -> ndarray:
        msg = """This is a fake forward function. It will be masked on initialization by _forward_i2c or _forward_cg"""
        raise NotImplementedError(f"Class \'AbstractPool2DLayerCPU\'. Error: {msg}")

    def backward(self, dy: ndarray) -> ndarray:
        msg = """This is a fake backward function. It will be masked on initialization by _backward_i2c or _backward_cg"""
        raise NotImplementedError(f"Class \'AbstractPool2DLayerCPU\'. Error: {msg}")
    # ---

    def _forward_nchw_cython(self, x: ndarray) -> ndarray:
        raise NotImplementedError()

    def _backward_nchw_cython(self, dy: ndarray) -> ndarray:
        raise NotImplementedError()

    def _forward_nhwc_cython(self, x: ndarray) -> ndarray:
        raise NotImplementedError()

    def _backward_nhwc_cython(self, dy: ndarray) -> ndarray:
        raise NotImplementedError()
