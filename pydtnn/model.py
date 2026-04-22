"""
PyDTNN model
"""
from pydtnn.context.show import Context_Show
from pydtnn.context.train import Context_Train
from pydtnn.utils.constants import Array
import logging
logger = logging.getLogger(__name__)


class Model[T: Array](Context_Train[T], Context_Show[T]):
    ...
