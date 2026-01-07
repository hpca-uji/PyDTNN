import numpy as np

from pydtnn.backends.cpu.losses.loss import LossCPU
from pydtnn.losses.kl_divergence import KLDivergence


class KLDivergenceCPU(KLDivergence[np.ndarray], LossCPU):

    def compute(self, y_pred, y_targ, batch_size):
        # loss = np.abs(y_targ * (np.log(np.abs(y_targ / (y_pred + self.eps)) + 1)))
        # loss = np.sum(loss) / y_pred.shape[0]
        # dx = - pred / target # Respecto a Target
        dx = np.log(np.abs(y_targ/(y_pred + self.eps)) + 1)  # Respecto a prediction
        dx = dx / batch_size
        loss = np.sum(dx)
        return loss, dx
