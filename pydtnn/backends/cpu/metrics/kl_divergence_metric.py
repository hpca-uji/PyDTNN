import numpy as np

from pydtnn.backends.cpu.metrics.metric import MetricCPU
from pydtnn.metrics.kl_divergence_metric import KLDivergenceMetric


class KLDivergenceMetricCPU(KLDivergenceMetric[np.ndarray], MetricCPU):

    def compute(self, y_pred, y_targ):
        loss = np.abs(y_pred * np.log(np.abs(y_pred / (y_targ + self.eps) + self.eps)))
        loss = np.sum(loss) / y_pred.shape[0]
        return loss
