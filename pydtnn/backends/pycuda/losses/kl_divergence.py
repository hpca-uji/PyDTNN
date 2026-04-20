from pydtnn.backends.pycuda.losses.loss import LossPycuda
from pydtnn.losses.kl_divergence import KLDivergence
from pydtnn.backends.pycuda.utils.tensor_array import TensorArray
from pycuda import gpuarray  # type: ignore
import numpy as np
import logging
logger = logging.getLogger(__name__)


class KLDivergencePycuda(KLDivergence[TensorArray], LossPycuda):

    def compute(self, y_pred, y_targ, batch_size):
        # loss = SUM(|pred * log(|pred / (targ + eps)| + eps) / N
        # dx = log(|pred / targ + eps| + eps) + 1 / N

        self.kernel(y_targ.ary, y_pred.ary, self.loss, self.dx.ary,
                    np.int32(self.model.batch_size), np.int32(batch_size),
                    np.int32(np.prod(self.shape[1:])), np.float32(self.eps),
                    grid=self.grid, block=self.block,
                    stream=self.model.stream)
        # loss = gpuarray.sum(self.loss).get()
        loss = gpuarray.sum(self.dx.ary).get()
        return loss, self.dx
