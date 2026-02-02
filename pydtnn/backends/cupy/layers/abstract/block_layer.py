from pydtnn.backends.cupy.layers.layer import LayerCUPY
from pydtnn.layers.abstract.block_layer import AbstractBlockLayer
import cupy as np


class AbstractBlockLayerCUPY(AbstractBlockLayer[np.ndarray], LayerCUPY):
    pass
