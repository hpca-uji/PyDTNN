"""Numpy backend implementation of the Kullback-Leibler Divergence loss."""

import logging
from typing import TYPE_CHECKING

from pydtnn.backends.numpy.losses.abstract.loss import LossNumpy
from pydtnn.libs import numpy as np
from pydtnn.losses.kl_divergence import KLDivergence

__all__ = ("KLDivergenceNumpy",)

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    import numpy as np  # noqa: F811 (override typing)


class KLDivergenceNumpy(KLDivergence[np.ndarray], LossNumpy):
    """Numpy-based implementation of the KL Divergence loss function."""

    def _model_init(self) -> None:
        """Initializes the model and updates memory usage tracking."""
        super()._model_init()
        self.memory_used += self.dx.nbytes

    def compute(self, y_pred: np.ndarray, y_targ: np.ndarray, batch_size: int) -> tuple[float, np.ndarray]:
        """
        Computes the KL Divergence loss and the gradient with respect to predictions.

        Args:
            y_pred: Predicted values.
            y_targ: Target values.
            batch_size: Size of the current batch.

        Returns:
            A tuple containing the scalar loss value and the gradient array.
        """
        # loss = np.abs(y_targ * (np.log(np.abs(y_targ / (y_pred + self.eps)) + 1)))
        # loss = np.sum(loss) / y_pred.shape[0]
        # dx = - pred / target # Respecto a Target

        # dx = np.log(np.abs(y_targ/(y_pred + self.eps)) + 1)  # Respecto a prediction
        # dx = dx / batch_size
        dx = self.dx[: y_targ[0]]

        np.add(y_pred, self.eps, out=dx)
        np.divide(y_targ, dx, out=dx)
        np.abs(dx, out=dx)
        np.add(dx, 1, out=dx)
        np.log(dx, out=dx)
        np.divide(dx, batch_size, out=dx)

        loss: np.ndarray = np.sum(dx)
        return loss.item(), dx
