import numpy as np

from pydtnn.metrics.metric import Metric


class RegressionMSE(Metric):

    def __call__(self, y_pred, y_targ):
        return np.square(y_targ - y_pred).mean()
