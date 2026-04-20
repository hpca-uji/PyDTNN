from pydtnn.utils.constants import DTYPE2CTYPE, ArrayShape
from typing import TYPE_CHECKING
from pydtnn.libs import numpy as np
from pydtnn.backends.numpy.layers.batch_normalization import BatchNormalizationNumpy
from pydtnn.backends.cupy.layers.layer import LayerCupy
import logging
logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    import numpy as np


class BatchNormalizationCupy(BatchNormalizationNumpy, LayerCupy):

    def _model_init(self, prev_shape: ArrayShape, x: np.ndarray | None = None) -> None:
        super()._model_init(prev_shape, x)

        self.stream_2 = np.cuda.Stream()
        self.defines_replaces = {"\"TYPE\"": DTYPE2CTYPE[self.model.dtype]}

        self.bwd = self._bwd_kernel()
        # ----

    def _training_bwd(self, dx: np.ndarray, dy: np.ndarray) -> None:
        # return super()._training_bwd(dx, dy)
        dim_i, dim_j = dx.shape
        self.bwd(self.model.cuda_grid,
                 self.model.cuda_block,
                 (dx, dy, self.xn,
                  self.std, self.gamma,
                  self.dgamma, self.dbeta,
                  dim_i, dim_j, dx.size))
    # ---
