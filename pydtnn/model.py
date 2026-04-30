
from pydtnn.model_context.print import Print
from pydtnn.model_context.train import Train
from pydtnn.utils.constants import Array


class Model[T: Array](Train[T], Print[T]):
    """
    # PyDTNN model
    """
