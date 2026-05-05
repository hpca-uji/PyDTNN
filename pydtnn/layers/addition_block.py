import logging

from pydtnn.layers.abstract.block_layer import AbstractBlockLayer
from pydtnn.utils.constants import Array

__all__ = (
    "AdditionBlock",
)

logger = logging.getLogger(__name__)


class AdditionBlock[T: Array](AbstractBlockLayer[T]):
    pass
