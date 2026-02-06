from pydtnn.libs import numpy as np
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    import numpy as np

from pydtnn.backends.numpy.losses.loss import LossNumpy
from pydtnn.losses.kl_divergence import KLDivergence


class KLDivergenceNumpy(KLDivergence[np.ndarray], LossNumpy):

    def initialize(self) -> None:
        super().initialize()
        self.real_memory_size += self.dx.nbytes

    def compute(self, y_pred, y_targ, batch_size):
        # loss = np.abs(y_targ * (np.log(np.abs(y_targ / (y_pred + self.eps)) + 1)))
        # loss = np.sum(loss) / y_pred.shape[0]
        # dx = - pred / target # Respecto a Target
        # ----

        # dx = np.log(np.abs(y_targ/(y_pred + self.eps)) + 1)  # Respecto a prediction
        # dx = dx / batch_size
        dx = self.dx[:y_targ[0]]

        np.add(y_pred, self.eps, out=dx)
        np.divide(y_targ, dx, out=dx)
        np.abs(dx, out=dx)
        np.add(dx, 1, out=dx)
        np.log(dx, out=dx)
        np.divide(dx, batch_size, out=dx)

        loss = float(np.sum(dx))
        return loss, dx
