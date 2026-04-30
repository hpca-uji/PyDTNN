
from pydtnn.model.repr import Repr
from pydtnn.model.train import Train
from pydtnn.utils.constants import Array


class Model[T: Array](Train[T], Repr[T]):
    """
    # PyDTNN model
    """
