import numpy as np

from pydtnn.backends.cpu.metrics.binary_confusion_matrix import BinaryConfusionMatrixCPU
from pydtnn.backends.cpu.metrics.metric import MetricCPU
from pydtnn.metrics.precision import Precision
from pydtnn.backends.cpu.utils.div_arrays_set_if_zero import div_arrays_set_if_zero


class PrecisionCPU(Precision[np.ndarray], MetricCPU):

    conf_matrix_metric: BinaryConfusionMatrixCPU

    def compute(self, y_pred: np.ndarray, y_targ: np.ndarray) -> float:
        true_positives = self.conf_matrix_metric.get_true_positives()
        false_positives = self.conf_matrix_metric.get_false_positives()
        # true_positives / (true_positives + false_positives)
        precision = np.asarray(true_positives, dtype=np.dtype(float), order="C", copy=True)
        divider = np.add(true_positives, false_positives, dtype=np.dtype(float), order="C")
        div_arrays_set_if_zero(precision,  divider, default_value=0.0)
        return float(np.average(precision))