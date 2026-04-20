from pydtnn.backends.pycuda.utils.tensor_array import TensorArray
from pydtnn.backends.pycuda.losses.loss import LossPycuda
from pydtnn.losses.binary_cross_entropy import BinaryCrossEntropy
from pycuda import gpuarray  # type: ignore
import logging
logger = logging.getLogger(__name__)


class BinaryCrossEntropyPycuda(LossPycuda, BinaryCrossEntropy[TensorArray]):

    def compute(self, y_pred: TensorArray, y_targ: TensorArray, batch_size: int) -> tuple[float, TensorArray]:

        assert len(y_targ.shape) == 2
        self.kernel(y_targ, y_pred, self.loss, self.dx.ary,
                    batch_size, self.shape[1], self.eps,
                    grid=self.grid, block=self.block,
                    stream=self.model.stream)
        loss: float = -gpuarray.sum(self.loss[:batch_size]) / batch_size
        return loss, self.dx
