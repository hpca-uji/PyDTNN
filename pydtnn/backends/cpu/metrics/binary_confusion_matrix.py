from pydtnn.backends.cpu.metrics.metric import MetricCPU
from pydtnn.metrics.binary_confusion_matrix import BinaryConfusionMatrix

from pydtnn.libs import numpy as np

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
        False: TRUE_NEGATIVE
    },
    False: {
        # bool(y_targ[i, label])
        True: FALSE_NEGATIVE,
        False: FALSE_POSITIVE
    }
}


class BinaryConfusionMatrixCPU(BinaryConfusionMatrix[np.ndarray], MetricCPU):

    def initialize(self) -> None:
        super().initialize()
        _, target_classes = self.shape
        self.conf_matrix = np.zeros((target_classes, 2, 2), dtype=np.int32)

        self.real_memory_size += self.conf_matrix.nbytes
    # ---

    def compute(self, y_pred: np.ndarray, y_targ: np.ndarray) -> np.ndarray:
        """
        For every label in target class, the output will have one confusion matrix like this:
                |Predicted|
        ________| T  | F  |
        Target|T| TP | FN |
              |F| FP | TN |
        """
        y_targ = np.asarray(y_targ, dtype=self.model.dtype)
        # NOTE: y_pred.shape == y_targ.shape == (n<=self.model.batch_size, self.model.output_shape)
        n, target_classes = y_pred.shape
        # assert target_classes == pred_classes, f"target_classes ({target_classes}) != pred_classes {pred_classes}, and must have the same value."
        self.conf_matrix.fill(0)

        for i in range(n):
            for label in range(target_classes):
                expected = bool(np.isclose(y_targ[i, label], y_pred[i, label], rtol=1e-03, atol=1e-3))
                is_positive = bool(y_targ[i, label])
                self.conf_matrix[label, *(_dict_indexes[expected][is_positive])] += 1

        return self.conf_matrix

    def get_true_positives(self):
        return self.conf_matrix[:, *TRUE_POSITIVE]

    def get_true_negatives(self):
        return self.conf_matrix[:, *TRUE_NEGATIVE]

    def get_false_positives(self):
        return self.conf_matrix[:, *FALSE_NEGATIVE]

    def get_false_negatives(self):
        return self.conf_matrix[:, *FALSE_POSITIVE]
