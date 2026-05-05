import logging

from pydtnn.backends.direct.abstract.layerable import LayerableDirect
from pydtnn.backends.numpy.layers.layer import LayerNumpy

__all__ = (
    "LayerDirect",
)

logger = logging.getLogger(__name__)


class LayerDirect(LayerNumpy, LayerableDirect):
    ...
