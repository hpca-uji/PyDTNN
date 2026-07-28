"""
Dataset module for PyDTNN.

Provides the base Dataset class and utility functions for managing,
transforming, and generating data batches for machine learning models.
"""

from __future__ import annotations

import logging
from enum import IntEnum
from typing import TYPE_CHECKING
from collections.abc import Generator, Callable

import numpy as np

from pydtnn.utils.constants import ArrayShape

__all__ = ("Base",)

logger = logging.getLogger(__name__)


if TYPE_CHECKING:
    from pydtnn.model.utils import Utils as Model


type TransformFunc = Callable[[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]


class Base:
    """
    Abstract base class for all datasets in PyDTNN.

    Defines the interface and common utilities for data partitioning,
    shape management, and format conversion.
    """

    class Part(IntEnum):
        """Enum representing the dataset partition."""

        TRAIN = 0
        VAL = 1
        TEST = 2

    model: Model
    debug: bool
    test_as_validation: bool
    x_empty_batch: np.ndarray[tuple[int, ...]]
    y_empty_batch: np.ndarray[tuple[int, ...]]
    input_shape: ArrayShape
    output_shape: ArrayShape
    class_weight: list[float] = []
    _nsamples: list[int]
    _augments: dict[Base.Part, list[TransformFunc]]
    _initial_nsamples: list[int]
    _local_offset: list[int]
    _local_nsamples: list[int]
    _local_remaining_nsamples: list[int]
    _x: list[np.ndarray[tuple[int, ...]]]
    _y: list[np.ndarray[tuple[int, ...]]]
    _data_generator: Callable[[Base.Part], Generator[tuple[np.ndarray, np.ndarray]]]
