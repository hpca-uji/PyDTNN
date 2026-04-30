import logging
from typing import TYPE_CHECKING

from cupy.cuda import Stream  # type: ignore

from pydtnn.backends.cupy.layers.layer import LayerCupy
from pydtnn.backends.numpy.layers.batch_normalization import \
    BatchNormalizationNumpy
from pydtnn.utils.constants import DTYPE2CTYPE, ArrayShape

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    import numpy as np


class BatchNormalizationCupy(BatchNormalizationNumpy, LayerCupy):

    def _model_init(self, prev_shape: ArrayShape, x: np.ndarray | None = None) -> None:
        super()._model_init(prev_shape, x)

        self.stream_2 = Stream()
        self.defines_replaces = {"\"TYPE\"": DTYPE2CTYPE[self.model.dtype]}

        self.fwd = self._fwd_kernel()
        self.bwd = self._bwd_kernel()

    def _training_fwd(self, x: np.ndarray, _mean: np.ndarray, _var: np.ndarray, y: np.ndarray) -> None:
        # return super()._training_bwd(dx, dy)
        dim_i, dim_j = x.shape
        self.fwd(self.model.cuda_grid,
                 self.model.cuda_block,
                 (x, y, self.xn,
                  self.std, self.gamma,
                  self.beta, _mean,
                  _var, self.epsilon,
                  dim_i, dim_j, x.size))

    def _training_bwd(self, dx: np.ndarray, dy: np.ndarray) -> None:
        # return super()._training_bwd(dx, dy)
        dim_i, dim_j = dx.shape
        self.bwd(self.model.cuda_grid,
                 self.model.cuda_block,
                 (dx, dy, self.xn,
                  self.std, self.gamma,
                  self.dgamma, self.dbeta,
                  dim_i, dim_j, dx.size))
