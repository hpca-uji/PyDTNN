import logging

from pydtnn.backends.cython.abstract.layerable import LayerableCython
from pydtnn.backends.numpy.activations.activation import ActivationNumpy

__all__ = (
    "ActivationCython",
)

logger = logging.getLogger(__name__)


class ActivationCython(ActivationNumpy, LayerableCython):
    ...
