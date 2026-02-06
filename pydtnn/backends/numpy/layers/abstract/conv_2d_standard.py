from pydtnn.libs import numpy as np
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    import numpy as np
from pydtnn.backends.numpy.layers.abstract.conv_2d import AbstractConv2DNumpy
from pydtnn.utils.tensor import TensorFormat, format_transpose
from pydtnn.utils.constants import Parameters


class AbstractConv2DStandardNumpy(AbstractConv2DNumpy):
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

    def _export_prop(self, key: str):
        if key not in {Parameters.WEIGHTS, Parameters.DW}:
            return super()._export_prop(key)
        value = getattr(self, key)

        match self.model.tensor_format:
            case TensorFormat.NHWC:
                # NHWC's src: ci, kh, kw, co
                # NCHW's dst: co, ci, kh, kw
                return np.asarray(format_transpose(value, "IHWO", "OIHW"), dtype=np.float64).copy()
        return super()._export_prop(key)
    # -----

    def _import_prop(self, key: str, value) -> None:
        if key not in {Parameters.WEIGHTS, Parameters.DW}:
            return super()._import_prop(key, value)

        match self.model.tensor_format:
            case TensorFormat.NHWC:
                # NCHW's src: co, ci, kh, kw
                # NHWC's dst: ci, kh, kw, co
                ary = getattr(self, key)
                ary[:] = np.asarray(format_transpose(value, "OIHW", "IHWO"), dtype=self.model.dtype)
                return
        return super()._import_prop(key, value)
    # -----
