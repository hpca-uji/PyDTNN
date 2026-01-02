from pydtnn.metrics.metric import Metric
from pydtnn.metrics.binary_confusion_matrix import BinaryConfusionMatrix
from pydtnn.utils.constants import Array


class F1Score[T: Array](Metric[T]):
    order = BinaryConfusionMatrix.order + 1
    conf_matrix_metric: BinaryConfusionMatrix = None  # type: ignore
    format = "f1: %.4f"

    def initialize(self) -> None:

        for metric in self.model.metrics_funcs:
            if isinstance(metric, BinaryConfusionMatrix):
                self.conf_matrix_metric = metric
                break
        assert self.conf_matrix_metric is not None, f"F1-Score requires of {BinaryConfusionMatrix.__name__}"
        super().initialize()
