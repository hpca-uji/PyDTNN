from pydtnn.backends.cupy.metrics.metric import MetricCupy
from pydtnn.backends.numpy.metrics.multiclass_confusion_matrix import MulticlassConfusionMatrixNumpy

import cupy as np


class MulticlassConfusionMatrixCupy(MulticlassConfusionMatrixNumpy, MetricCupy):

    def compute(self, y_pred: np.ndarray, y_targ: np.ndarray) -> np.ndarray:
        return super().compute(y_pred, y_targ).get()
