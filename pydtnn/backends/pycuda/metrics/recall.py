import logging

import numpy as np

from pydtnn.backends.pycuda.metrics.metric import MetricPycuda
from pydtnn.backends.pycuda.utils.tensor_array import TensorArray
from pydtnn.metrics.recall import Recall

__all__ = (
    "RecallPycuda",
)

logger = logging.getLogger(__name__)


class RecallPycuda(Recall[TensorArray], MetricPycuda):
    def _model_init(self) -> None:
        super()._model_init()
        target_classes = self.model.output_shape[0]
        self.recall = TensorArray.new_zeros(shape=(1,), dtype=self.model.dtype, tensor_format=self.model.tensor_format, cudnn_dtype=self.model.cudnn_dtype)
        self.local_recall = TensorArray.new_zeros(shape=(target_classes,), dtype=self.model.dtype, tensor_format=self.model.tensor_format, cudnn_dtype=self.model.cudnn_dtype)

    def compute(self, y_pred: TensorArray, y_targ: TensorArray) -> float:

        target_classes = self.model.output_shape[0]

        self.recall.fill(0)
        self.local_recall.fill(0)

        target_classes = np.int32(target_classes)
        self.kernel(self.recall.ary, self.conf_matrix_metric.conf_matrix.ary, self.local_recall.ary, target_classes, grid=self.grid, block=self.block, stream=self.model.stream)

        return float(self.recall.get())
