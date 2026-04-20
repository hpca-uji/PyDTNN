from pydtnn.utils.constants import Array
from pydtnn.metrics.binary_confusion_matrix import BinaryConfusionMatrix
from pydtnn.metrics.metric import Metric
import logging
logger = logging.getLogger(__name__)


class Precision[T: Array](Metric[T]):
    order = BinaryConfusionMatrix.order + 1
    conf_matrix_metric: BinaryConfusionMatrix[T] = None  # type: ignore
    format = "prec: %.4f"

    def _model_init(self) -> None:
        super()._model_init()
        for metric in self.model.metrics_funcs:
            if isinstance(metric, BinaryConfusionMatrix):
                self.conf_matrix_metric = metric
                break
        assert self.conf_matrix_metric is not None, f"Precision requires of {BinaryConfusionMatrix.__name__}"
