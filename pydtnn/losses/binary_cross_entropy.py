from pydtnn.utils.constants import Array
from pydtnn.losses.loss import Loss
import logging
logger = logging.getLogger(__name__)


class BinaryCrossEntropy[T: Array](Loss[T]):
    format = "bce: %.7f"
