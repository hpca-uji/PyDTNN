from pydtnn.backends.numpy.abstract.layerable import LayerableNumpy
from pydtnn.backends.winograd.abstract.base import BaseWinograd


class LayerableWinograd(LayerableNumpy, BaseWinograd):
    ...
