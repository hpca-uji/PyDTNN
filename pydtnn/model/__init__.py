"""PyDTNN model module providing the base Model class for the framework"""

from pydtnn.model.train import Train
from pydtnn.utils.constants import Array

__all__ = ("Model",)


class Model[T: Array](Train[T]):  # noqa: D101 (generics not detected)
    """
    # PyDTNN model
    The Model class serves as the primary interface for PyDTNN, integrating
    training, inference, state management, and representation capabilities.
    """
