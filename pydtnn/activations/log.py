import logging

from pydtnn.activations.activation import Activation
from pydtnn.utils.constants import Array

__all__ = ("Log",)

logger = logging.getLogger(__name__)


class Log[T: Array](Activation[T]):
    # NOTE: It is a LogSigmoid activation
    pass
