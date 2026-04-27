import logging
from typing import TYPE_CHECKING

from pydtnn.backends.cython.layers.layer import LayerCython
from pydtnn.backends.cython.utils.bn_training_cython import (
    bn_training_bwd_cython, bn_training_fwd_cython)
from pydtnn.backends.numpy.layers.batch_normalization import \
    BatchNormalizationNumpy
from pydtnn.libs import numpy as np

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    import numpy as np


class BatchNormalizationCython(BatchNormalizationNumpy, LayerCython):

    def _training_fwd(self, x: np.ndarray, _mean: np.ndarray, _var: np.ndarray, y: np.ndarray) -> None:
        bn_training_fwd_cython(x, y, self.xn, self.std, self.gamma, self.beta, _mean, _var, self.epsilon)  # type: ignore
    # ---

    def _training_bwd(self, dx: np.ndarray, dy: np.ndarray) -> None:
        bn_training_bwd_cython(dx, dy, self.xn, self.std, self.gamma, self.dgamma, self.dbeta)  # type: ignore
    # ---
