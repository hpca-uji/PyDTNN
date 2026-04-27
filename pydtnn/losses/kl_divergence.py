import logging

from pydtnn.losses.loss import Loss
from pydtnn.utils.constants import Array

logger = logging.getLogger(__name__)


class KLDivergence[T: Array](Loss[T]):
    format = "kld: %.7f"
