import numpy as np

from pydtnn.backends.cpu.metrics.metric import MetricCPU
from pydtnn.metrics.kl_divergence_metric import KLDivergenceMetric


class KLDivergenceMetricCPU(KLDivergenceMetric[np.ndarray], MetricCPU):

    def initialize(self) -> None:
        super().initialize()
        self.loss = np.ndarray(self.shape, dtype=self.model.dtype, order="C")

        self.actual_size += self.loss.size

    def compute(self, y_pred: np.ndarray, y_targ: np.ndarray) -> float:
        loss = self.loss[:y_pred.shape[0]]
        # loss = np.abs(y_pred * np.log(np.abs(y_pred / (y_targ + eps) + eps)))
        np.add(y_targ, self.eps, out=loss)
        np.divide(y_pred, loss, out=loss)
        np.add(loss, self.eps, out=loss)
        np.abs(loss, out=loss)
        np.log(loss, out=loss)
        np.multiply(y_pred, loss, out=loss)
        np.abs(loss, out=loss)
        
        loss = np.sum(loss) / y_pred.shape[0]
        return loss
