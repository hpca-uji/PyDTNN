from pydtnn.losses.loss import Loss
from pydtnn.utils.constants import Array


class BinaryCrossEntropy[T: Array](Loss[T]):
    format = "bce: %.7f"
