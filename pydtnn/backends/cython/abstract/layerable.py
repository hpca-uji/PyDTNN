from pydtnn.backends.cython.abstract.base import BaseCython
from pydtnn.backends.numpy.abstract.layerable import LayerableNumpy

__all__ = ("LayerableCython",)


class LayerableCython(LayerableNumpy, BaseCython):
    ...
