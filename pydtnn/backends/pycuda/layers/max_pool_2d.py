from pydtnn.utils.constants import ArrayShape
from pydtnn.backends.pycuda.utils.tensor_array import TensorArray
from pydtnn.libs import cudnn as cudnn
from pydtnn.backends.pycuda.layers.abstract.pool_2d_layer import AbstractPool2DLayerPycuda
from pydtnn.layers.max_pool_2d import MaxPool2D
import logging
logger = logging.getLogger(__name__)


class MaxPool2DPycuda(MaxPool2D[TensorArray], AbstractPool2DLayerPycuda):

    def _model_init(self, prev_shape: ArrayShape, x: TensorArray) -> None:
        super()._model_init(prev_shape, x)
        pool_mode = cudnn.cudnnPoolingMode['CUDNN_POOLING_MAX']
        self.initialize_pool_2d_gpu(prev_shape, x, pool_mode)
