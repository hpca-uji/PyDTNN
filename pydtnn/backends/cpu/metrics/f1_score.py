import numpy as np

from pydtnn.backends.cpu.metrics.metric import MetricCPU
from pydtnn.backends.cpu.metrics.binary_confusion_matrix import BinaryConfusionMatrixCPU
from pydtnn.metrics.f1_score import F1Score
from pydtnn.backends.cpu.utils.div_arrays_set_if_zero import div_arrays_set_if_zero

class F1ScoreCPU(F1Score[np.ndarray], MetricCPU):

    conf_matrix_metric: BinaryConfusionMatrixCPU

    def initialize(self) -> None:
        super().initialize()
        self.true_positives = np.zeros(self.model.batch_size, dtype=np.float32, order="C")
        self.false_positives = np.zeros(self.model.batch_size, dtype=np.float32, order="C")
        self.false_negatives = np.zeros(self.model.batch_size, dtype=np.float32, order="C")
        self.are_zeros = np.zeros(self.model.batch_size, dtype=np.bool, order="C")

        self.actual_size += self.true_positives.size + self.false_positives.size + self.false_negatives.size + self.are_zeros.size
    # ----

    def compute(self, y_pred: np.ndarray, y_targ: np.ndarray) -> float:
        n = y_pred.shape[0]
        true_positives = self.true_positives[:n]
        false_positives = self.false_positives[:n]
        false_negatives = self.false_negatives[:n]
        are_zeros = self.are_zeros[:n]

        # This variable is not necessary, is to make the code more understandable.
        aggregation = false_positives
        f1 = aggregation

        np.copyto(true_positives, self.conf_matrix_metric.get_true_positives())
        np.copyto(false_positives, self.conf_matrix_metric.get_false_positives())
        np.copyto(false_negatives, self.conf_matrix_metric.get_false_negatives())

        # f1 =  2 * true_positives / (2 * true_positives + false_positives + false_negatives
        np.multiply(2, true_positives, out=true_positives)
        np.add(true_positives, false_positives, out=aggregation)
        np.add(aggregation, false_negatives, out=aggregation)

        #div_arrays_set_if_zero(true_positives,  aggregation, default_value=0.0)
        np.not_equal(aggregation, 0, out=are_zeros)
        np.divide(true_positives, aggregation, out=f1, where=(are_zeros))

        return float(np.average(f1))
