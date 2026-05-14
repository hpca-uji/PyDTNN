"""
PyCUDA implementation of the Kullback-Leibler Divergence loss function.
"""

import logging

import numpy as np
from pycuda import gpuarray  # type: ignore

from pydtnn.backends.pycuda.losses.loss import LossPycuda
from pydtnn.backends.pycuda.utils.tensor_array import TensorArray
from pydtnn.losses.kl_divergence import KLDivergence

__all__ = ("KLDivergencePycuda",)

logger = logging.getLogger(__name__)


class KLDivergencePycuda(KLDivergence[TensorArray], LossPycuda):
    """
    PyCUDA-accelerated Kullback-Leibler Divergence loss calculation.
    """

    def compute(self, y_pred, y_targ, batch_size):
        """
        Computes the KL Divergence loss and its gradient on the GPU.

        Args:
            y_pred (TensorArray): Predicted probability distribution.
            y_targ (TensorArray): Target probability distribution.
            batch_size (int): Number of samples in the current batch.

        Returns:
            tuple: A tuple containing the scalar loss value and the gradient TensorArray.
        """
        # loss = SUM(|pred * log(|pred / (targ + eps)| + eps) / N
        # dx = log(|pred / targ + eps| + eps) + 1 / N

        self.kernel(
            y_targ.ary,
            y_pred.ary,
            self.loss,
            self.dx.ary,
            np.int32(self.model.batch_size),
            np.int32(batch_size),
            np.int32(np.prod(self.shape[1:])),
            np.float32(self.eps),
            grid=self.grid,
            block=self.block,
            stream=self.model.stream,
        )
        # loss = gpuarray.sum(self.loss).get()
        loss = gpuarray.sum(self.dx.ary).get()
        return loss.item(), self.dx
