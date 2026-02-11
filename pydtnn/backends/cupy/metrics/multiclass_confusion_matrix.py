from pydtnn.backends.numpy.metrics.multiclass_confusion_matrix import MulticlassConfusionMatrixNumpy

import cupy as np


class MulticlassConfusionMatrixCUPY(MulticlassConfusionMatrixNumpy):

    def compute(self, y_pred: np.ndarray, y_targ: np.ndarray) -> np.ndarray:
        return super().compute(y_pred, y_targ).get()
