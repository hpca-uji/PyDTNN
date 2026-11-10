"""Setted Output layer implementation."""

import logging

from pydtnn.layers.abstract.layer import Layer
from pydtnn.utils.constants import Array, ArrayShape
from pydtnn.utils.initializers import InitializerFunc, zeros

__all__ = ("ForwardedOutput",)

logger = logging.getLogger(__name__)


class ForwardedOutput[T: Array](Layer[T]):  # noqa: D101 (generics not detected)
    """A layer that return a previously setted value."""

    def __init__(
        self,
        shape: ArrayShape = (1,),
        forward_initializer: InitializerFunc = zeros,
        backward_initializer: InitializerFunc = zeros,
        use_fwd_seted_value: bool = True,
        use_bwd_seted_value: bool = True
    ) -> None:
        """
        Initializes the Scalar layer.

        Args:
            shape: The shape of the input data.
        """
        super().__init__(shape)
        self.forward_initializer = forward_initializer
        self.backward_initializer = backward_initializer

        if use_fwd_seted_value:
            self.forward = self.forward_seted_value

        if use_bwd_seted_value:
            self.backward = self.backward_seted_value

        # The following attributes will be setted during the execution
        self.seted_forward_output: T = None  # pyright: ignore[reportAttributeAccessIssue]
        self.seted_backward_output: T = None  # pyright: ignore[reportAttributeAccessIssue]

    def _model_init(self, prev_shape: ArrayShape, x: T | None) -> None:
        super()._model_init(prev_shape, x)
        self.shape = prev_shape

    def set_forward_output(self, new_output: T) -> None:
        """Method to set the forward's output."""
        self.seted_forward_output = new_output

    def set_backward_output(self, new_output: T) -> None:
        """Method to set the backward's output."""
        self.seted_backward_output = new_output

    def forward(self, x: T) -> T:
        return x

    def backward(self, dy: T) -> T:
        return dy

    def forward_seted_value(self, x: T) -> T:
        return self.seted_forward_output

    def backward_seted_value(self, dy: T) -> T:
        return self.seted_backward_output
