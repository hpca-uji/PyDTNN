from pydtnn.backends.cpu.metrics.metric import MetricCPU
from pydtnn.metrics.multiclass_confusion_matrix import MulticlassConfusionMatrix

import numpy as np

class MulticlassConfusionMatrixCPU(MulticlassConfusionMatrix[np.ndarray], MetricCPU):

    def compute(self, y_pred: np.ndarray, y_targ: np.ndarray) -> np.ndarray:
        """
        The output will be a confusion matrix like this:
                |Predicted     |
        ________| 0  | 1  | 2  |
        Target|0| T0 | F1 | F2 |
              |1| F0 | T1 | F2 |
              |2| F0 | F1 | T2 |
        """
        
        # NOTE: y_pred.shape == y_targ.shape == (n<=self.model.batch_size, self.model.output_shape)
        n, target_classes = y_pred.shape
        #assert target_classes == pred_classes, f"target_classes ({target_classes}) != pred_classes {pred_classes}, and must have the same value."
        conf_matrix = np.zeros((target_classes, target_classes), dtype=np.int32)

        for i in range(n):
            target_class = np.nonzero(y_targ[i]== 1)[0]
            predicted_class = np.nonzero(y_pred[i]== 1)[0]
            conf_matrix[target_class, predicted_class] += 1
        
        return conf_matrix
