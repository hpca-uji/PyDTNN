from pydtnn.backends.cupy.metrics.metric import MetricCupy
from pydtnn.backends.numpy.metrics.f1_score import F1ScoreNumpy
from pydtnn.libs import numpy as np
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    import numpy as np

class F1ScoreCupy(F1ScoreNumpy, MetricCupy):

    def compute(self, y_pred: np.ndarray, y_targ: np.ndarray) -> float:
        true_positives = self.true_positives
        false_positives = self.false_positives
        false_negatives = self.false_negatives

        # This variable is not necessary, is to make the code more understandable.
        aggregation = false_positives
        f1 = aggregation

        true_positives[:] = self.conf_matrix_metric.get_true_positives()
        false_positives[:] = self.conf_matrix_metric.get_false_positives()
        false_negatives[:] = self.conf_matrix_metric.get_false_negatives()

        # f1 =  2 * true_positives / (2 * true_positives + false_positives + false_negatives
        np.multiply(2, true_positives, out=true_positives)
        np.add(true_positives, false_positives, out=aggregation)
        np.add(aggregation, false_negatives, out=aggregation)

        for i in range(true_positives.shape[0]):
            f1[i] = (true_positives[i] / aggregation[i]) if aggregation[i] != 0 else 0

        return float(np.average(f1))
