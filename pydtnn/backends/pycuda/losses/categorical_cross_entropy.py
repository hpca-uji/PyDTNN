from pydtnn.backends.pycuda.utils.tensor_array import TensorArray
from pydtnn.backends.pycuda.losses.loss import LossPycuda
from pydtnn.losses.categorical_cross_entropy import CategoricalCrossEntropy
from pycuda import gpuarray  # type: ignore
import numpy as np
import logging
logger = logging.getLogger(__name__)

class CategoricalCrossEntropyPycuda(LossPycuda, CategoricalCrossEntropy[TensorArray]):

    def compute(self, y_pred: TensorArray, y_targ: TensorArray, batch_size: int) -> tuple[float, TensorArray]:
        self.kernel(y_targ.ary, y_pred.ary, self.loss, self.dx.ary,
                    np.int32(batch_size), np.int32(self.shape[1]), np.float32(self.eps),
                    grid=self.grid, block=self.block,
                    stream=self.model.stream)
        loss: float = -gpuarray.sum(self.loss[:batch_size]).get() / batch_size
        return loss, self.dx
