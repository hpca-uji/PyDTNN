"""Numpy backend implementation for multiclass confusion matrix calculation."""

import logging
from typing import TYPE_CHECKING

from pydtnn.backends.numpy.metrics.abstract.metric import MetricNumpy
from pydtnn.libs import numpy as np
from pydtnn.metrics.multiclass_confusion_matrix import MulticlassConfusionMatrix

__all__ = ("MulticlassConfusionMatrixNumpy",)


logger = logging.getLogger(__name__)


if TYPE_CHECKING:
    import numpy as np  # noqa: F811 (override typing)


class MulticlassConfusionMatrixNumpy(MulticlassConfusionMatrix[np.ndarray], MetricNumpy):
    """Numpy-based multiclass confusion matrix metric."""

    def _model_init(self) -> None:
        """Initializes the confusion matrix buffer based on model output shape."""
        super()._model_init()
        _, target_classes = self.shape
        self.conf_matrix: np.ndarray = np.zeros((target_classes, target_classes), dtype=np.int32)
        self.memory_used += self.conf_matrix.nbytes

    def compute(self, y_pred: np.ndarray, y_targ: np.ndarray) -> np.ndarray:
        """
        Computes the confusion matrix for a given batch of predictions and targets.

        Args:
            y_pred: Predicted labels in one-hot encoded format.
            y_targ: Ground truth labels in one-hot encoded format.

        Returns:
            A 2D numpy array representing the confusion matrix.

        The output will be a confusion matrix like this:
                |Predicted     |
        ________| 0  | 1  | 2  |
        Target|0| T0 | F1 | F2 |
              |1| F0 | T1 | F2 |
              |2| F0 | F1 | T2 |
        """

        b = self.model.real_batch_size
        y_targ = np.asarray(y_targ[:b], dtype=self.model.dtype, order="C")

        # NOTE: y_pred.shape == y_targ.shape == (n<=self.model.batch_size, self.model.output_shape)
        # assert target_classes == pred_classes, f"target_classes ({target_classes}) != pred_classes {pred_classes},"
        # " and must have the same value."
        # conf_matrix = np.zeros((target_classes, target_classes), dtype=np.int32)
        self.conf_matrix.fill(0)

        for i in range(b):
            target_class = np.nonzero(y_targ[i] == 1)[0]
            predicted_class = np.nonzero(y_pred[i] == 1)[0]
            self.conf_matrix[target_class, predicted_class] += 1

        return self.conf_matrix
