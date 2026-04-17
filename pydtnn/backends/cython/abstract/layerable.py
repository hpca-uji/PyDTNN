from pydtnn.backends.cython.abstract.base import BaseCython
from pydtnn.backends.numpy.abstract.layerable import LayerableNumpy


class LayerableCython(LayerableNumpy, BaseCython):
    ...
