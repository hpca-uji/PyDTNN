"""Utilities for performing 0231 tensor transposition using various optimized backends."""

import logging
from collections.abc import Callable

import numpy as np

from pydtnn.utils.best_of.best_of import BestOf
from pydtnn.utils.transpose_cython import transpose_0231_ijk_cython, transpose_0231_ikj_cython

__all__ = (
    "transpose_0231_ijk_cython_wrapper",
    "transpose_0231_ikj_cython_wrapper",
    "transpose_0231_numpy",
)

logger = logging.getLogger(__name__)


def transpose_0231_numpy(
    original: np.ndarray,
    transposed: np.ndarray | None = None,
) -> np.ndarray:
    """
    Perform 0231 transposition using NumPy's transpose method.

    Args:
        original: The input 4D array.
        transposed: Optional pre-allocated output array.

    Returns:
        The transposed array.
    """
    d0, d1, d2, d3 = original.shape
    if transposed is None:
        transposed = np.empty((d0, d2, d3, d1), original.dtype)
    transposed[...] = original.transpose((0, 2, 3, 1))
    return transposed


def transpose_0231_ijk_cython_wrapper(
    original: np.ndarray,
    transposed: np.ndarray | None = None,
) -> np.ndarray:
    """
    Perform 0231 transposition using Cython implementation with ijk loop order.

    Args:
        original: The input 4D array.
        transposed: Optional pre-allocated output array.

    Returns:
        The transposed array.
    """
    d0, d1, d2, d3 = original.shape
    if transposed is None:
        transposed = np.empty((d0, d2, d3, d1), original.dtype)
    transpose_0231_ijk_cython(original, transposed)
    return transposed


def transpose_0231_ikj_cython_wrapper(
    original: np.ndarray,
    transposed: np.ndarray | None = None,
) -> np.ndarray:
    """
    Perform 0231 transposition using Cython implementation with ikj loop order.

    Args:
        original: The input 4D array.
        transposed: Optional pre-allocated output array.

    Returns:
        The transposed array.
    """
    d0, d1, d2, d3 = original.shape
    if transposed is None:
        transposed = np.empty((d0, d2, d3, d1), original.dtype)
    transpose_0231_ikj_cython(original, transposed)
    return transposed


# TODO: change typing "Callable, etc."
best_transpose_0231: Callable[[np.ndarray, np.ndarray | None], np.ndarray] = BestOf(
    name="Transpose 0231 methods",
    alternatives=[
        ("numpy", transpose_0231_numpy),
        ("ikj_cyt", transpose_0231_ikj_cython_wrapper),
        ("ijk_cyt", transpose_0231_ijk_cython_wrapper),
    ],
    get_problem_size=lambda *args: args[0].shape,
)
