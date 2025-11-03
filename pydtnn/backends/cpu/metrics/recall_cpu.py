import numpy as np

from pydtnn.backends.cpu.metrics.binary_confusion_matrix_cpu import BinaryConfusionMatrixCPU
from pydtnn.backends.cpu.metrics.metric_cpu import MetricCPU
from pydtnn.metrics.recall import Recall


class RecallCPU(MetricCPU, Recall[np.ndarray]):

    conf_matrix_metric: BinaryConfusionMatrixCPU

    def compute(self, y_pred: np.ndarray, y_targ: np.ndarray) -> float:
        true_positives = self.conf_matrix_metric.get_true_positives()
        false_negatives = self.conf_matrix_metric.get_false_negatives()

        return float(np.average(true_positives / (true_positives + false_negatives)))