import logging

from pydtnn.layers.abstract.conv_2d import AbstractConv2D
logger = logging.getLogger(__name__)

from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from pydtnn.activations.activation import Activation
from pydtnn.layers.layer import Layer
from pydtnn.utils.initializers import InitializerFunc, glorot_uniform, zeros
from pydtnn.utils.constants import Array, ArrayShape, Parameters

import math

class Conv2D[T: Array](AbstractConv2D[T]):
    ...
