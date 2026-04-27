import logging

from pydtnn.losses.loss import Loss
from pydtnn.utils.constants import Array

logger = logging.getLogger(__name__)


class CategoricalCrossEntropy[T: Array](Loss[T]):
    format = "cce: %.7f"
