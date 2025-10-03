from abc import ABC, abstractmethod

from pydtnn.backends import PromoteToBackendMixin
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from pydtnn.model import Model
from numpy import ndarray
from pydtnn.backends.gpu import TensorGPU
type Array = ndarray | TensorGPU


class Loss(PromoteToBackendMixin, ABC):

    def __init__(self, shape: tuple[int, ...], model: "Model", eps=1e-8):
        self.shape = shape
        self.model = model
        self.eps = eps

    @abstractmethod
    def __call__(self, y_pred: Array, y_targ: Array, batch_size: int) -> tuple[float, Array]:
        pass
