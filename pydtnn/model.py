
from pydtnn.session.debug import Debug
from pydtnn.session.train import Train
from pydtnn.utils.constants import Array


class Model[T: Array](Train[T], Debug[T]):
    """
    # PyDTNN model
    """
