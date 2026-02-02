from pydtnn.libs import libnumpy as np

from pydtnn.backends.cpu.metrics.binary_confusion_matrix import BinaryConfusionMatrixCPU
from pydtnn.backends.cpu.metrics.metric import MetricCPU
from pydtnn.metrics.recall import Recall
# from pydtnn.backends.cpu.utils.div_arrays_set_if_zero import div_arrays_set_if_zero


class RecallCPU(Recall[np.ndarray], MetricCPU):

    conf_matrix_metric: BinaryConfusionMatrixCPU

    def initialize(self) -> None:
        super().initialize()
        self.temp_var_shape = (self.shape[1], )
        self.temp_memory_size += int(2 * np.prod(self.temp_var_shape)) * np.float32().itemsize
        self.temp_memory_size += int(1 * np.prod(self.temp_var_shape)) * np.bool().itemsize
        self.real_memory_size += self.temp_memory_size
    # ----

    def post_initialize(self) -> None:
        super().post_initialize()
        self.true_positives = self.model.memory.ndarray(self.temp_var_shape, dtype=np.float32)
        self.false_negatives = self.model.memory.ndarray(self.temp_var_shape, dtype=np.float32)
        self.are_zeros = self.model.memory.ndarray(self.temp_var_shape, dtype=np.bool)
        self.model.memory.free(self.temp_memory_size)

    def compute(self, y_pred: np.ndarray, y_targ: np.ndarray) -> float:
        y_targ = np.asarray(y_targ, dtype=self.model.dtype)
        true_positives = self.true_positives
        false_negatives = self.false_negatives
        are_zeros = self.are_zeros
        # This two variables are not necessary, are to make the code more understandable.
        real_positives = false_negatives
        recall = false_negatives

        np.copyto(true_positives, self.conf_matrix_metric.get_true_positives())
        np.copyto(true_positives, self.conf_matrix_metric.get_false_negatives())
        # true_positives / (true_positives + false_negatives)
        np.add(true_positives, false_negatives, dtype=np.dtype(float), out=real_positives)
        # div_arrays_set_if_zero(recall,  divider, default_value=0.0)

        np.not_equal(real_positives, 0, out=are_zeros)
        np.divide(true_positives, real_positives, out=recall, where=(are_zeros))
        return float(np.average(recall))
