from pydtnn.losses.loss import Loss
from pydtnn.utils.constants import Array


class KLDivergence[T: Array](Loss[T]):
    format = "kld: %.7f"
