from pydtnn.backends.cpu.layers.conv_2d import Conv2DCPU
from pydtnn.utils.tensor import TensorFormat

class Conv2DStandardCPU(Conv2DCPU):
    # NOTE: This is an abstract class.

    def _initializing_special_parameters(self):
        super()._initializing_special_parameters()
        match self.model.tensor_format:
                case TensorFormat.NCHW:
                    self.weights_shape = (self.co, self.ci, *self.filter_shape)
                case TensorFormat.NHWC:
                    self.weights_shape = (self.ci, *self.filter_shape, self.co)
                case _:
                    raise NotImplementedError(f"\"{self.model.tensor_format}\" format not implemented.")
    # ---