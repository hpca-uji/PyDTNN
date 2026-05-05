from pydtnn.model.repr import Repr
from pydtnn.model.train import Train
from pydtnn.utils.constants import Array

__all__ = ("Model",)


class Model[T: Array](Train[T], Repr[T]):
    """
    # PyDTNN model
    """
