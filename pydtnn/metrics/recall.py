from pydtnn.metrics.metric import Metric
from pydtnn.metrics.binary_confusion_matrix import BinaryConfusionMatrix
from pydtnn.utils.types import Array


class Recall[T: Array](Metric[T]):
    order = BinaryConfusionMatrix.order + 1
    conf_matrix_metric: BinaryConfusionMatrix = None  # type: ignore

    def initialize(self) -> None:

        for metric in self.model.metrics_funcs:
            if isinstance(metric, BinaryConfusionMatrix):
                self.conf_matrix_metric = metric
                break
        assert self.conf_matrix_metric is not None, f"Recall requires of {BinaryConfusionMatrix.__name__}"
        super().initialize()
