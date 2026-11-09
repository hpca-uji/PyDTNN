"""Scalar layer implementation for PyDTNN."""

import logging
from typing import Any

from pydtnn.abstract.layerable import Layerable
from pydtnn.activations.softmax import Softmax
from pydtnn.layers.identity import Identity
from pydtnn.layers.multiplication import Multiplication
from pydtnn.layers.scalar import Scalar
from pydtnn.utils.constants import Array

__all__ = ("ScaledDotProductAttention",)

logger = logging.getLogger(__name__)


class ScaledDotProductAttention[T: Array](Multiplication[T]):  # noqa: D101 (generics not detected)
    """Implementation of a Scaled Dot-Product Attention. https://arxiv.org/pdf/1706.03762, Figure 2 (left)."""

    def __init__(
            self,
            mult_layers: tuple[list[list[Layerable[T]]], Layerable[T]] = (
                [[Identity[T]()], [Identity[T]()]],
                Identity[T]()
            ),
            with_masked_layer: bool = False,
            **kwargs: Any
    ) -> None:
        """
        Parameters:
            - mult_layers (tuple[list[list[Layerable[T]]], Layerable[T]]): The multiplication's input layers.
                By default, they will be a "[[Identity()], [Identity()]]" for the first Multiplication and a Identity() for the last one
                (that is, they will be executed without any modifications).
        """
        first_path: list[Layerable[T]] = [
            Multiplication(
                paths=[*mult_layers[0]],
                multiple_inputs=True
            ),
            Scalar(),
        ]
        if with_masked_layer:
            first_path.append(
                Identity()  # TODO: Implement the Mask layer.
            )
        # else: nothing special.
        first_path.append(Softmax())

        paths: list[list[Layerable[T]]] = [
            first_path,
            [
                mult_layers[1]
            ]
        ]

        super().__init__(paths, multiple_inputs=True, **kwargs)
