"""
PyDTNN model
"""
from pydtnn._model.model_show import Model_Show
from pydtnn._model.model_train import Model_Train
from pydtnn.utils.constants import Array
import logging
logger = logging.getLogger(__name__)


class Model[T: Array](Model_Train[T], Model_Show[T]):
    ...
