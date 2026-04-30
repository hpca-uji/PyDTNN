
from pydtnn.model_parts.print import Print
from pydtnn.model_parts.train import Train
from pydtnn.utils.constants import Array


class Model[T: Array](Train[T], Print[T]):
    """
    # PyDTNN model
    """
