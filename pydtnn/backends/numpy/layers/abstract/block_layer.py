import logging
logger = logging.getLogger(__name__)

from pydtnn.backends.numpy.layers.layer import LayerNumpy
from pydtnn.layers.abstract.block_layer import AbstractBlockLayer
from pydtnn.libs import numpy as np
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    import numpy as np


class AbstractBlockLayerNumpy(AbstractBlockLayer[np.ndarray], LayerNumpy):
    pass
