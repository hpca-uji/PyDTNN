import logging
logger = logging.getLogger(__name__)

from pydtnn.backends.numpy.activations.activation import ActivationNumpy


class ActivationCython(ActivationNumpy):
    ...
