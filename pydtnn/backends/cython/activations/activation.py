from pydtnn.backends.numpy.activations.activation import ActivationNumpy
from pydtnn.backends.cython.abstract.layerable import LayerableCython
import logging
logger = logging.getLogger(__name__)


class ActivationCython(ActivationNumpy, LayerableCython):
    ...
