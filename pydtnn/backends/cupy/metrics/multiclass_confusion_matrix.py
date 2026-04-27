import numpy as np
from pydtnn.backends.numpy.metrics.multiclass_confusion_matrix import MulticlassConfusionMatrixNumpy
from pydtnn.backends.cupy.metrics.metric import MetricCupy
import logging
logger = logging.getLogger(__name__)


class MulticlassConfusionMatrixCupy(MulticlassConfusionMatrixNumpy, MetricCupy):

    def compute(self, y_pred: np.ndarray, y_targ: np.ndarray) -> np.ndarray:
        return np.asarray(super().compute(y_pred, y_targ))
