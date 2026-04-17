import logging
logger = logging.getLogger(__name__)

from pydtnn.backends.cython.abstract.layerable import LayerableCython
from pydtnn.backends.numpy.activations.activation import ActivationNumpy


class ActivationCython(ActivationNumpy, LayerableCython):
    ...
