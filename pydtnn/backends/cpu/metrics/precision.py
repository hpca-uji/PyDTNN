from pydtnn.libs import numpy as np

from pydtnn.backends.cpu.metrics.binary_confusion_matrix import BinaryConfusionMatrixCPU
from pydtnn.backends.cpu.metrics.metric import MetricCPU
from pydtnn.metrics.precision import Precision
# from pydtnn.backends.cpu.utils.div_arrays_set_if_zero import div_arrays_set_if_zero


class PrecisionCPU(Precision[np.ndarray], MetricCPU):

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
        self.false_positives = self.model.memory.ndarray(self.temp_var_shape, dtype=np.float32)
        self.are_zeros = self.model.memory.ndarray(self.temp_var_shape, dtype=np.bool)
        self.model.memory.free(self.temp_memory_size)

    def compute(self, y_pred: np.ndarray, y_targ: np.ndarray) -> float:
        y_targ = np.asarray(y_targ, dtype=self.model.dtype)
        true_positives = self.true_positives
        false_positives = self.false_positives
        are_zeros = self.are_zeros
        # This two variables are not necessary, are to make the code more understandable.
        positives = false_positives
        precision = false_positives

        np.copyto(true_positives, self.conf_matrix_metric.get_true_positives())
        np.copyto(false_positives, self.conf_matrix_metric.get_false_positives())
        # true_positives / (true_positives + false_positives)

        np.add(true_positives, false_positives, out=positives)
        # precision = (precision / divider if divider[i] != 0 else default_value)
        # div_arrays_set_if_zero(precision,  f_positives, default_value=0)
        np.not_equal(positives, 0, out=are_zeros)
        np.divide(true_positives, positives, out=precision, where=(are_zeros))
        return float(np.average(precision))
