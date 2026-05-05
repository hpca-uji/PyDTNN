from pydtnn.backends.direct.abstract.base import BaseDirect
from pydtnn.backends.numpy.abstract.layerable import LayerableNumpy

__all__ = (
    "LayerableDirect",
)


class LayerableDirect(LayerableNumpy, BaseDirect):
    ...
