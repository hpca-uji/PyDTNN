from pydtnn.utils.constants import Array
from pydtnn.losses.loss import Loss
import logging
logger = logging.getLogger(__name__)


class KLDivergence[T: Array](Loss[T]):
    format = "kld: %.7f"
