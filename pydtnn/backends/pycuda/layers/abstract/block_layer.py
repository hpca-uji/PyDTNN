from pydtnn.layers.abstract.block_layer import AbstractBlockLayer
from pydtnn.backends.pycuda.utils.tensor_array import TensorArray
from pydtnn.backends.pycuda.layers.layer import LayerPycuda
import logging
logger = logging.getLogger(__name__)


class AbstractBlockLayerPycuda(AbstractBlockLayer[TensorArray], LayerPycuda):
    pass
