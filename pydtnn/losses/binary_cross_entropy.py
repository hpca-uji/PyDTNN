import logging

from pydtnn.losses.loss import Loss
from pydtnn.utils.constants import Array

__all__ = (
    "BinaryCrossEntropy",
)

logger = logging.getLogger(__name__)


class BinaryCrossEntropy[T: Array](Loss[T]):
    format = "bce: %.7f"
