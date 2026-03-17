import logging
logger = logging.getLogger(__name__)

from pydtnn.backends.pycuda.layers.layer import LayerPycuda
from pydtnn.backends.pycuda.utils.tensor_array import TensorArray
from pydtnn.layers.abstract.block_layer import AbstractBlockLayer


class AbstractBlockLayerPycuda(AbstractBlockLayer[TensorArray], LayerPycuda):
    pass
