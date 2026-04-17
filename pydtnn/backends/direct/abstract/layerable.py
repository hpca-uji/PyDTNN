from pydtnn.backends.direct.abstract.base import BaseDirect
from pydtnn.backends.numpy.abstract.layerable import LayerableNumpy


class LayerableDirect(LayerableNumpy, BaseDirect):
    ...
