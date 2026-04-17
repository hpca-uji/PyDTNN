import numpy as np

from pydtnn.abstract.layerable import Layerable
from pydtnn.backends.numpy.abstract.base import BaseNumpy


class LayerableNumpy(Layerable[np.ndarray], BaseNumpy):
    ...
