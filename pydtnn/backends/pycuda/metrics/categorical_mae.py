import logging
logger = logging.getLogger(__name__)

import numpy as np

from pydtnn.metrics.categorical_mae import CategoricalMAE

from pydtnn.backends.pycuda.metrics.metric import MetricPycuda
from pydtnn.backends.pycuda.utils.tensor_array import TensorArray

class CategoricalMAEPycuda(CategoricalMAE[TensorArray], MetricPycuda):

    def _model_init(self) -> None:
        super()._model_init()
        self.res = TensorArray.new_zeros(shape=(1, ), dtype=np.dtype(self.model.dtype),
                                         tensor_format=self.model.tensor_format, cudnn_dtype=self.model.cudnn_dtype)
        self.local_res = TensorArray.new_zeros(shape=(self.model.batch_size, ), dtype=np.dtype(self.model.dtype),
                                               tensor_format=self.model.tensor_format, cudnn_dtype=self.model.cudnn_dtype)

    def compute(self, y_pred: TensorArray, y_targ: TensorArray) -> float:
        n = y_pred.shape[0]

        self.res.fill(0)
        self.local_res.fill(0)

        n = np.int32(n)
        num_classes = np.int32(y_pred.shape[1])

        self.kernel(y_targ.ary, y_pred.ary,
                    self.res.ary, self.local_res.ary,
                    n, num_classes,
                    grid=self.grid, block=self.block,
                    stream=self.model.stream)
        return self.res.ary.get()[0]
