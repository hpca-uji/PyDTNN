import numpy as np

from pydtnn.layers.layer import Layer, LayerError
from pydtnn.utils.constants import Array


class AdaptiveAveragePool2D[T: Array](Layer):

    # This layer will calculate the pool shape and the stride from the output shape (passed as parameter) and the previous layer shape.

    # output_shape:
    #  -> None: if the output shape is equal to the input
    #  -> int: if all the output shape's dimensions share values
    #  -> Tuple[int, int]: if it is necessary or it is preferred to define each output dimension individually

    def __init__(self, output_shape: int | tuple[int, int] | None = None):
        super().__init__()
        self.output_shape = output_shape

        # This value will change in initialize:
        self.pooling_not_needed: bool = None  # type: ignore
    # ---  END __init__ --- #

    def initialize(self, prev_shape: tuple[int, int], x: T | None) -> None:
        super().initialize(prev_shape, x)

        self.ci, self.hi, self.wi = self.model.decode_shape(prev_shape)

        if self.output_shape is None:
            self.ho, self.wo = self.hi, self.wi
        else:
            self.ho, self.wo = (self.output_shape, self.output_shape) if isinstance(self.output_shape, int) else self.output_shape

        if not (self.ho > 0 and self.wo > 0):
            raise LayerError(f"The output height and width should be grater than 0. height: {self.ho} width: {self.wo}")
        self.co = self.ci

        # If the output and the input shapes are the same, there is no need of pooling.
        self.pooling_not_needed = (self.hi == self.ho) and (self.wi == self.wo)

        self.shape = self.model.encode_shape((self.co, self.ho, self.wo))
        self.n = np.prod(self.shape)
    # - END initialize - #
