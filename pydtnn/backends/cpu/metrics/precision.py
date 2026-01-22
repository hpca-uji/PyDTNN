import numpy as np

from pydtnn.backends.cpu.metrics.binary_confusion_matrix import BinaryConfusionMatrixCPU
from pydtnn.backends.cpu.metrics.metric import MetricCPU
from pydtnn.metrics.precision import Precision
#from pydtnn.backends.cpu.utils.div_arrays_set_if_zero import div_arrays_set_if_zero


class PrecisionCPU(Precision[np.ndarray], MetricCPU):

    conf_matrix_metric: BinaryConfusionMatrixCPU

    def initialize(self) -> None:
        super().initialize()
        self.true_positives = np.zeros(self.model.batch_size, dtype=np.float32, order="C")
        self.false_positives = np.zeros(self.model.batch_size, dtype=np.float32, order="C")
        self.are_zeros = np.zeros(self.model.batch_size, dtype=np.bool, order="C")

        self.actual_size += self.true_positives.size + self.false_positives.size + self.are_zeros.size
    # ----

    def compute(self, y_pred: np.ndarray, y_targ: np.ndarray) -> float:
        n = y_pred.shape[0]
        true_positives = self.true_positives[:n]
        false_positives = self.false_positives[:n]
        are_zeros = self.are_zeros[:n]
        # This two variables are not necessary, are to make the code more understandable.
        positives = false_positives
        precision = false_positives

        np.copyto(true_positives, self.conf_matrix_metric.get_true_positives())
        np.copyto(false_positives, self.conf_matrix_metric.get_false_positives())
        # true_positives / (true_positives + false_positives)

        np.add(true_positives, false_positives, out=positives)
        # precision = (precision / divider if divider[i] != 0 else default_value)
        #div_arrays_set_if_zero(precision,  f_positives, default_value=0)
        np.not_equal(positives, 0, out=are_zeros)
        np.divide(true_positives, positives, out=precision, where=(are_zeros))
        return float(np.average(precision))