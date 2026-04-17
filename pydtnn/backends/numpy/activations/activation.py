from typing import TYPE_CHECKING
from pydtnn.backends.numpy.abstract.layerable import LayerableNumpy
from pydtnn.libs import numpy as np
from pydtnn.activations.activation import Activation
import logging
logger = logging.getLogger(__name__)


try:
    from pydtnn.libs.mpi import MPI
except Exception:
    pass
if TYPE_CHECKING:
    import numpy as np


class ActivationNumpy(Activation[np.ndarray], LayerableNumpy):
    ...
