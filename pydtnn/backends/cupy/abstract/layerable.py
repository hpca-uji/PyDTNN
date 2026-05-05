from pydtnn.backends.cupy.abstract.base import BaseCupy
from pydtnn.backends.numpy.abstract.layerable import LayerableNumpy

__all__ = ("LayerableCupy",)


class LayerableCupy(LayerableNumpy, BaseCupy): ...
