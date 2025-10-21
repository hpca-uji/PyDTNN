from pydtnn.backends.cpu.layers import LayerCPU
from pydtnn.layers import BatchNormalizationRelu
from pydtnn.cython import bn_relu_inference_cython
from pydtnn.model import Model
from pydtnn.utils.tensor import PYDTNN_TENSOR_FORMAT
from pydtnn.utils.best_transpose_0231 import best_transpose_0231
from pydtnn.utils.best_transpose_0312 import best_transpose_0312

from numpy import ndarray, empty, asarray


class BatchNormalizationReluCPU(LayerCPU, BatchNormalizationRelu):

    def initialize(self, prev_shape, x = None):
        super().initialize(prev_shape, x)
        self.y = empty(shape=(self.model.batch_size, *self.shape), dtype=self.model.dtype, order="C")

    def forward(self, x: ndarray) -> ndarray:
        """Version of the forward function that uses the BN + Relu"""

        if self.model.mode is Model.Mode.TRAIN:
            raise SystemExit("Sorry, fused layers cannot be used in training mode!")

        if self.spatial:
            if self.model.tensor_format is PYDTNN_TENSOR_FORMAT.NCHW:
                x = best_transpose_0231(x)
            x = x.reshape((-1, self.ci), copy=False, order="C")

        y:ndarray = self.y[: x.shape[0], :]
        bn_relu_inference_cython(x, 
                                 y.reshape((-1, self.ci), copy=False, order="C"), 
                                 self.running_mean, 
                                 self.inv_std, 
                                 self.gamma, 
                                 self.beta)

        if self.spatial:
            y = y.reshape((-1, self.hi, self.wi, self.ci), copy=False)
            if self.model.tensor_format is PYDTNN_TENSOR_FORMAT.NCHW:
                y = best_transpose_0312(y)
        return asarray(y, dtype=self.model.dtype, order='C', copy=None)

    def backward(self, x: ndarray) -> ndarray:
        raise SystemExit(f"Backward method of {self.__class__.__name__} should not be called")
