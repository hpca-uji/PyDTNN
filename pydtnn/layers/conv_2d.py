import logging
import math
from typing import TYPE_CHECKING, Optional

from pydtnn.layers.abstract.conv_2d import AbstractConv2D
from pydtnn.layers.layer import Layer
from pydtnn.utils.constants import Array, ArrayShape, Parameters
from pydtnn.utils.initializers import InitializerFunc, glorot_uniform, zeros

__all__ = (
    "Conv2D",
)

logger = logging.getLogger(__name__)


if TYPE_CHECKING:
    from pydtnn.activations.activation import Activation


class Conv2D[T: Array](AbstractConv2D[T]):
    ...
