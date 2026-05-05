from pydtnn.backends.numpy.abstract.layerable import LayerableNumpy
from pydtnn.backends.winograd.abstract.base import BaseWinograd

__all__ = (
    "LayerableWinograd",
)


class LayerableWinograd(LayerableNumpy, BaseWinograd):
    ...
