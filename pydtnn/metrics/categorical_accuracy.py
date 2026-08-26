"""Categorical accuracy metric implementation for PyDTNN."""

import logging
from typing import Any

from pydtnn.metrics.abstract.metric import Metric
from pydtnn.utils.constants import Array

__all__ = ("CategoricalAccuracy",)

logger = logging.getLogger(__name__)


class CategoricalAccuracy[T: Array](Metric[T]):  # noqa: D101 (generics not detected)
    """Metric to calculate the categorical accuracy of model predictions."""

    def format(self, value: Any) -> str:
        name = super().format(value).split(":", 1)[0]
        return f"{name}: {value * 100:.2f}%"
