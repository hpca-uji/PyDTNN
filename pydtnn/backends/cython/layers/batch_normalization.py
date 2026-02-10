from pydtnn.backends.numpy.layers.batch_normalization import BatchNormalizationNumpy
from pydtnn.libs import numpy as np
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    import numpy as np
from pydtnn.backends.cython.utils.bn_training_cython import bn_training_bwd_cython , bn_training_fwd_cython


class BatchNormalizationCython(BatchNormalizationNumpy):

    #def _training_fwd(self, x: np.ndarray, _mean: np.ndarray, _var: np.ndarray, y: np.ndarray) -> None:
    #    bn_training_fwd_cython(x, y, self.xn, self.std, self.gamma, self.beta, _mean, _var, self.epsilon)
    # ---

    def _training_bwd(self, dx: np.ndarray, dy: np.ndarray) -> None:
        bn_training_bwd_cython(dx, dy, self.xn, self.std, self.gamma, self.dgamma, self.dbeta)
    # ---
