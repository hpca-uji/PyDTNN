import logging

import numpy as np

from pydtnn.backends.cupy.metrics.metric import MetricCupy
from pydtnn.backends.numpy.metrics.binary_confusion_matrix import \
    BinaryConfusionMatrixNumpy

logger = logging.getLogger(__name__)


class BinaryConfusionMatrixCupy(BinaryConfusionMatrixNumpy, MetricCupy):

    def compute(self, y_pred: np.ndarray, y_targ: np.ndarray) -> np.ndarray:
        return np.asarray(super().compute(y_pred, y_targ))
