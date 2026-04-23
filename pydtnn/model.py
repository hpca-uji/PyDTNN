
from pydtnn.context.show import Show
from pydtnn.context.train import Train
from pydtnn.utils.constants import Array


class Model[T: Array](Train[T], Show[T]):
    """
    # PyDTNN model
    """
    # Context Inheritance:
    # Base - Layer - Init - Reduce - Eval - Train - Model
    #      \                                     /
    #       --------------- Show ----------------
