from typing import Self
from pydtnn.activations.activation import Activation
from pydtnn.utils.types import ArrayShape, Array

class Relu[T: Array](Activation[T]):
    

    def __init__(self, shape: ArrayShape = (1,)):
        super().__init__(shape)
        # Will be initalized in "initialize"
        self.mask: T = None # type: ignore
    
    def copy_from(self, other: Self) -> None:
        super().copy_from(other)
        self.mask = other.mask.copy()  # typing: ignore (It's correct)
