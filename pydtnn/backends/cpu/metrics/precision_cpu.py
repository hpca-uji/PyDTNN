import numpy as np

from pydtnn.backends.cpu.metrics.binary_confusion_matrix_cpu import BinaryConfusionMatrixCPU
from pydtnn.backends.cpu.metrics.metric_cpu import MetricCPU
from pydtnn.metrics.precision import Precision


class PrecisionCPU(MetricCPU, Precision[np.ndarray]):

    conf_matrix_metric: BinaryConfusionMatrixCPU

    def compute(self, y_pred: np.ndarray, y_targ: np.ndarray) -> float:
        true_positives = self.conf_matrix_metric.get_true_positives()
        false_positives = self.conf_matrix_metric.get_false_positives()

        return float(np.average(true_positives / (true_positives + false_positives)))