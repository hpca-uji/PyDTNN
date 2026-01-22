import numpy as np

from pydtnn.backends.cpu.metrics.binary_confusion_matrix import BinaryConfusionMatrixCPU
from pydtnn.backends.cpu.metrics.metric import MetricCPU
from pydtnn.metrics.recall import Recall
#from pydtnn.backends.cpu.utils.div_arrays_set_if_zero import div_arrays_set_if_zero

class RecallCPU(Recall[np.ndarray], MetricCPU):

    conf_matrix_metric: BinaryConfusionMatrixCPU

    def initialize(self) -> None:
        super().initialize()
        self.true_positives = np.zeros(self.model.batch_size, dtype=np.float32, order="C")
        self.false_negatives = np.zeros(self.model.batch_size, dtype=np.float32, order="C")
        self.are_zeros = np.zeros(self.model.batch_size, dtype=np.bool, order="C")

        self.actual_size += self.true_positives.size + self.false_negatives.size + self.are_zeros.size
    # ----

    def compute(self, y_pred: np.ndarray, y_targ: np.ndarray) -> float:
        n = y_pred.shape[0]
        true_positives = self.true_positives[:n]
        false_negatives = self.false_negatives[:n]
        are_zeros = self.are_zeros[:n]
        # This two variables are not necessary, are to make the code more understandable.
        real_positives = false_negatives
        recall = false_negatives

        np.copyto(true_positives, self.conf_matrix_metric.get_true_positives())
        np.copyto(true_positives, self.conf_matrix_metric.get_false_negatives())
        # true_positives / (true_positives + false_negatives)
        np.add(true_positives, false_negatives, dtype=np.dtype(float), order="C", out=real_positives)
        #div_arrays_set_if_zero(recall,  divider, default_value=0.0)
        
        np.not_equal(real_positives, 0, out=are_zeros)
        np.divide(true_positives, real_positives, out=recall, where=(are_zeros))
        return float(np.average(recall))
