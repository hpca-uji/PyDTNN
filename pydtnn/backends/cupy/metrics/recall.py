import logging
logger = logging.getLogger(__name__)

from pydtnn.backends.cupy.metrics.metric import MetricCupy
from pydtnn.backends.numpy.metrics.recall import RecallNumpy
from pydtnn.libs import numpy as np
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    import numpy as np

class RecallCupy(RecallNumpy, MetricCupy):

    def _post_init(self) -> None:
        super()._post_init()
        with self.model.memory:
            self.true_positives = self.model.memory.ndarray(self.temp_var_shape, dtype=np.float32)
            self.false_positives = self.model.memory.ndarray(self.temp_var_shape, dtype=np.float32)
            self.false_negatives = self.model.memory.ndarray(self.temp_var_shape, dtype=np.float32)

    def compute(self, y_pred: np.ndarray, y_targ: np.ndarray) -> float:
        true_positives = self.true_positives
        false_negatives = self.false_negatives
        # This two variables are not necessary, are to make the code more understandable.
        real_positives = false_negatives
        recall = false_negatives

        np.copyto(true_positives, self.conf_matrix_metric.get_true_positives())
        np.copyto(true_positives, self.conf_matrix_metric.get_false_negatives())
        # true_positives / (true_positives + false_negatives)
        np.add(true_positives, false_negatives, dtype=np.dtype(float), out=real_positives)
        # div_arrays_set_if_zero(recall,  divider, default_value=0.0)

        for i in range(true_positives.shape[0]):
            recall[i] = (true_positives[i] / real_positives[i]) if real_positives[i] != 0 else 0
        return float(np.average(recall))
