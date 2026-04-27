
from pydtnn.session.show import Show
from pydtnn.session.train import Train
from pydtnn.utils.constants import Array


class Model[T: Array](Train[T], Show[T]):
    """
    # PyDTNN model
    """
    # Context Inheritance:
    # Base - Layer - Init - Reduce - Eval - Train - Model
    #      \                                     /
    #       --------------- Show ----------------
