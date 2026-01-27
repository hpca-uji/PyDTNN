import numpy as np

from pydtnn.backends.cpu.metrics.metric import MetricCPU
from pydtnn.backends.cpu.metrics.binary_confusion_matrix import BinaryConfusionMatrixCPU
from pydtnn.metrics.f1_score import F1Score
from pydtnn.backends.cpu.utils.div_arrays_set_if_zero import div_arrays_set_if_zero

class F1ScoreCPU(F1Score[np.ndarray], MetricCPU):

    conf_matrix_metric: BinaryConfusionMatrixCPU

    def initialize(self) -> None:
        super().initialize()
        shape = self.shape[1]

        self.temp_var_shape = (shape, )

        self.temp_memory_size = int(4 * np.prod(self.temp_var_shape))

        if not self.model.use_memory_pool:
            self.true_positives: np.ndarray = np.zeros(self.temp_var_shape, dtype=np.float32, order="C")
            self.false_positives: np.ndarray = np.zeros(self.temp_var_shape, dtype=np.float32, order="C")
            self.false_negatives: np.ndarray = np.zeros(self.temp_var_shape, dtype=np.float32, order="C")
            self.are_zeros: np.ndarray = np.zeros(self.temp_var_shape, dtype=np.bool, order="C")
        else:
            self.true_positives: np.ndarray = None  #type: ignore (It will be initialized later)
            self.false_positives: np.ndarray = None  #type: ignore (It will be initialized later)
            self.false_negatives: np.ndarray = None  #type: ignore (It will be initialized later)
            self.are_zeros: np.ndarray = None  #type: ignore (It will be initialized later)

        self.real_memory_size += self.temp_memory_size
    # ----

    def post_initialize(self) -> None:
        super().post_initialize()
        self.true_positives = np.asarray(self.model.memory_pool.get_ndarray(self.temp_var_shape), dtype=np.float32, order="C")
        self.false_positives = np.asarray(self.model.memory_pool.get_ndarray(self.temp_var_shape), dtype=np.float32, order="C")
        self.false_negatives = np.asarray(self.model.memory_pool.get_ndarray(self.temp_var_shape), dtype=np.float32, order="C")
        self.are_zeros = np.asarray(self.model.memory_pool.get_ndarray(self.temp_var_shape), dtype=np.bool, order="C")
        self.model.memory_pool.free_memory(self.temp_memory_size)

    def compute(self, y_pred: np.ndarray, y_targ: np.ndarray) -> float:
        true_positives = self.true_positives
        false_positives = self.false_positives
        false_negatives = self.false_negatives
        are_zeros = self.are_zeros

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
