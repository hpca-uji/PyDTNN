import numpy as np

from pydtnn.backends.cpu.metrics.metric import MetricCPU
from pydtnn.backends.cpu.metrics.binary_confusion_matrix import BinaryConfusionMatrixCPU
from pydtnn.metrics.f1_score import F1Score
from pydtnn.backends.cpu.utils.div_arrays_set_if_zero import div_arrays_set_if_zero

class F1ScoreCPU(F1Score[np.ndarray], MetricCPU):

    conf_matrix_metric: BinaryConfusionMatrixCPU

    def compute(self, y_pred: np.ndarray, y_targ: np.ndarray) -> float:
        true_positives = self.conf_matrix_metric.get_true_positives()
        false_positives = self.conf_matrix_metric.get_false_positives()
        false_negatives = self.conf_matrix_metric.get_false_negatives()

        # 2 * true_positives / (2 * true_positives + false_positives + false_negatives
        f1 = np.multiply(2, true_positives, dtype=np.dtype(float), order="C")
        divider = np.add(f1, false_positives,dtype=np.dtype(float), order="C")
        divider = np.add(divider, false_negatives, dtype=np.dtype(float), order="C")

        div_arrays_set_if_zero(f1,  divider, default_value=0.0)
        return float(np.average(f1))
