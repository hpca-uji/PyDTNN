from pydtnn.losses.loss import Loss
from pydtnn.utils.constants import Array


class CategoricalCrossEntropy[T: Array](Loss[T]):
    format = "cce: %.7f"
