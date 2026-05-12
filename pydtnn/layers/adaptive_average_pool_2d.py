"""
Adaptive Average Pooling 2D layer implementation for PyDTNN.
"""
import logging
import math

from pydtnn.layers.layer import Layer, LayerError
from pydtnn.utils.constants import Array, ArrayShape

__all__ = ("AdaptiveAveragePool2D",)

logger = logging.getLogger(__name__)


class AdaptiveAveragePool2D[T: Array](Layer):
    """
    Applies a 2D adaptive average pooling over an input signal composed of several input planes.

    The output size is fixed to the provided output_shape, regardless of the input size.
    """
    # This layer will calculate the pool shape and the stride from the output shape (passed as parameter) and the previous layer shape.
    # output_shape:
    #  -> None: if the output shape is equal to the input
    #  -> int: if all the output shape's dimensions share values
    #  -> Tuple[int, int]: if it is necessary or it is preferred to define each output dimension individually

    def __init__(self, output_shape: int | ArrayShape | None = None):
        """
        Initializes the AdaptiveAveragePool2D layer.

        Args:
            output_shape: The target output shape (H, W). If None, output shape equals input shape.
        """
        super().__init__()
        self.output_shape = output_shape

        # This value will change in initialize:
        self.pooling_not_needed: bool = None  # type: ignore

    def _model_init(self, prev_shape: ArrayShape, x: T | None) -> None:
        """
        Initializes layer parameters and calculates output dimensions.

        Args:
            prev_shape: The shape of the input tensor.
            x: Optional input tensor.
        """
        super()._model_init(prev_shape, x)

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
        self.n = math.prod(self.shape)

    @staticmethod
    def _index_first_element(index: int, dim_in: int, dim_out: int) -> int:
        """
        Calculates the starting index of the input window for a given output index.
        """
        return (index * dim_in) // dim_out

    @staticmethod
    def _index_last_element(index: int, dim_in: int, dim_out: int) -> int:
        """
        Calculates the ending index of the input window for a given output index.
        """
        return (((index + 1) * dim_in) + dim_out - 1) // dim_out