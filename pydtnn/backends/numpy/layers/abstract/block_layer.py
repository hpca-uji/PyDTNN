from typing import TYPE_CHECKING
from pydtnn.libs import numpy as np
from pydtnn.layers.abstract.block_layer import AbstractBlockLayer
from pydtnn.backends.numpy.layers.layer import LayerNumpy
import logging
logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    import numpy as np


class AbstractBlockLayerNumpy(AbstractBlockLayer[np.ndarray], LayerNumpy):
    pass
