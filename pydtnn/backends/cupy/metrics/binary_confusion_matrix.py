from pydtnn.backends.numpy.metrics.binary_confusion_matrix import BinaryConfusionMatrixNumpy

import cupy as np


class BinaryConfusionMatrixCUPY(BinaryConfusionMatrixNumpy):

    def compute(self, y_pred: np.ndarray, y_targ: np.ndarray) -> np.ndarray:
        return super().compute(y_pred, y_targ).get()
