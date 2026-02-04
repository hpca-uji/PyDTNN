from pydtnn.backends.cpu.layers.layer import LayerCPU
from pydtnn.layers.abstract.block_layer import AbstractBlockLayer
from pydtnn.libs import numpy as np
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    import numpy as np


class AbstractBlockLayerCPU(AbstractBlockLayer[np.ndarray], LayerCPU):
    pass
