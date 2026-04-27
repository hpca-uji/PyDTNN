import logging

from pydtnn.activations.activation import Activation
from pydtnn.utils.constants import Array

logger = logging.getLogger(__name__)


class Tanh[T: Array](Activation[T]):
    pass
