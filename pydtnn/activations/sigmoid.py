import logging
logger = logging.getLogger(__name__)

from pydtnn.activations.activation import Activation
from pydtnn.utils.constants import Array


class Sigmoid[T: Array](Activation[T]):
    pass
