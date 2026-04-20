from pydtnn.utils.constants import Array
from pydtnn.activations.activation import Activation
import logging
logger = logging.getLogger(__name__)


class Log[T: Array](Activation[T]):
    # NOTE: It is a LogSigmoid activation
    pass
