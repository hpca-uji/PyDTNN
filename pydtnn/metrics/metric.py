from abc import ABC, abstractmethod

from pydtnn.backends import PromoteToBackendMixin

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from model import Model

from pydtnn.utils.types import Array



class Metric(PromoteToBackendMixin, ABC):

    def __init__(self, shape: tuple[int, ...], model: "Model", eps=1e-8):
        self.shape = shape
        self.model = model
        self.eps = eps

    @abstractmethod
    def __call__(self, y_pred: Array, y_targ: Array) -> float:
        pass
