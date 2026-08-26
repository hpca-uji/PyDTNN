"""Numpy backend implementation for binary confusion matrix metrics."""

import logging
from typing import TYPE_CHECKING

from pydtnn.backends.numpy.metrics.abstract.metric import MetricNumpy
from pydtnn.libs import numpy as np
from pydtnn.metrics.binary_confusion_matrix import BinaryConfusionMatrix

__all__ = ("BinaryConfusionMatrixNumpy",)

logger = logging.getLogger(__name__)


if TYPE_CHECKING:
    import numpy as np  # noqa: F811 (override typing)

TRUE_POSITIVE = (0, 0)
TRUE_NEGATIVE = (1, 1)
FALSE_NEGATIVE = (0, 1)
FALSE_POSITIVE = (1, 0)

_dict_indexes = {
    # y_targ[i, label] == y_pred[i, label]
    # i.e.: "are both target' and prediciton' values the same?"
    True: {
        # bool(y_targ[i, label])
        # i.e.: "is the target' value 1 (True) or 0 (False)?"
        True: TRUE_POSITIVE,
        False: TRUE_NEGATIVE,
    },
    False: {
        # bool(y_targ[i, label])
        True: FALSE_NEGATIVE,
        False: FALSE_POSITIVE,
    },
}


class BinaryConfusionMatrixNumpy(BinaryConfusionMatrix[np.ndarray], MetricNumpy):
    """Numpy-based implementation of the binary confusion matrix metric."""

    def _model_init(self) -> None:
        """Initializes the confusion matrix buffer."""
        super()._model_init()
        _, target_classes = self.shape
        self.conf_matrix = np.zeros((target_classes, 2, 2), dtype=np.int32)

        self.memory_used += self.conf_matrix.nbytes

    def compute(self, y_pred: np.ndarray, y_targ: np.ndarray) -> np.ndarray:
        """
        Compute the binary confusion matrix metric.

        For every label in target class, the output will have one confusion matrix like this:
                |Predicted|
        ________| T  | F  |
        Target|T| TP | FN |
              |F| FP | TN |
        """

        b = self.model.real_batch_size
        y_targ = np.asarray(y_targ[:b], dtype=self.model.dtype, order="C")
        # NOTE: y_pred.shape == y_targ.shape == (n<=self.model.batch_size, self.model.output_shape)
        _, target_classes = y_pred.shape
        # assert target_classes == pred_classes, f"target_classes ({target_classes}) != pred_classes {pred_classes},"
        #                                           " and must have the same value."
        self.conf_matrix.fill(0)

        for i in range(b):
            for label in range(target_classes):
                expected = bool(
                    np.isclose(y_targ[i, label], y_pred[i, label], rtol=1e-03, atol=1e-3)
                )
                is_positive = bool(y_targ[i, label])
                self.conf_matrix[label, *(_dict_indexes[expected][is_positive])] += 1

        return self.conf_matrix

    def get_true_positives(self) -> np.ndarray:
        """Returns the count of true positives for each label."""
        return self.conf_matrix[:, *TRUE_POSITIVE]

    def get_true_negatives(self) -> np.ndarray:
        """Returns the count of true negatives for each label."""
        return self.conf_matrix[:, *TRUE_NEGATIVE]

    def get_false_positives(self) -> np.ndarray:
        """Returns the count of false positives for each label."""
        return self.conf_matrix[:, *FALSE_NEGATIVE]

    def get_false_negatives(self) -> np.ndarray:
        """Returns the count of false negatives for each label."""
        return self.conf_matrix[:, *FALSE_POSITIVE]
