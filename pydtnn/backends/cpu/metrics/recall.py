import numpy as np

from pydtnn.backends.cpu.metrics.binary_confusion_matrix import BinaryConfusionMatrixCPU
from pydtnn.backends.cpu.metrics.metric import MetricCPU
from pydtnn.metrics.recall import Recall
from pydtnn.backends.cpu.utils.div_arrays_set_if_zero import div_arrays_set_if_zero

class RecallCPU(Recall[np.ndarray], MetricCPU):

    conf_matrix_metric: BinaryConfusionMatrixCPU

    def compute(self, y_pred: np.ndarray, y_targ: np.ndarray) -> float:
        true_positives = self.conf_matrix_metric.get_true_positives()
        false_negatives = self.conf_matrix_metric.get_false_negatives()
        # true_positives / (true_positives + false_negatives)
        recall = np.asarray(true_positives, dtype=np.dtype(float), order="C")
        divider = np.add(true_positives, false_negatives, dtype=np.dtype(float), order="C")
        div_arrays_set_if_zero(recall,  divider, default_value=0.0)
        return float(np.average(recall))
