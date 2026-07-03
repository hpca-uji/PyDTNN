"""Binary Cross Entropy loss implementation for the PyDTNN framework."""

import logging

from pydtnn.losses.abstract.loss import Loss
from pydtnn.utils.constants import Array

__all__ = ("BinaryCrossEntropy",)

logger = logging.getLogger(__name__)


class BinaryCrossEntropy[T: Array](Loss[T]):  # noqa: D101 (generics not detected)
    """Computes the binary cross-entropy loss between target and output logits."""

    format = "bce: %.7f"
