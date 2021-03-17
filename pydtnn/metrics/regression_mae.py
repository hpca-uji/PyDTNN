import numpy as np

from pydtnn.metrics.metric import Metric


class RegressionMAE(Metric):

    def __call__(self, y_pred, y_targ):
        return np.sum(np.absolute(y_targ - y_pred))
