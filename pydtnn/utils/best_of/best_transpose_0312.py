"""Utilities for performing 0312 tensor transposition using various backend implementations."""

import logging
from collections.abc import Callable

import numpy as np

from pydtnn.utils.best_of.best_of import BestOf
from pydtnn.utils.transpose_cython import transpose_0312_ijk_cython, transpose_0312_ikj_cython

__all__ = (
    "transpose_0312_ijk_cython_wrapper",
    "transpose_0312_ikj_cython_wrapper",
    "transpose_0312_numpy",
)

logger = logging.getLogger(__name__)


def transpose_0312_numpy(
    original: np.ndarray,
    transposed: np.ndarray | None = None,
) -> np.ndarray:
    """
    Transpose a 4D array from (d0, d1, d2, d3) to (d0, d3, d1, d2) using NumPy.

    Args:
        original: The input 4D array.
        transposed: Optional pre-allocated output array.

    Returns:
        The transposed array.
    """
    d0, d1, d2, d3 = original.shape
    if transposed is None:
        transposed = np.empty((d0, d3, d1, d2), original.dtype)
    transposed[...] = original.transpose((0, 3, 1, 2))
    return transposed


def transpose_0312_ijk_cython_wrapper(
    original: np.ndarray,
    transposed: np.ndarray | None = None,
) -> np.ndarray:
    """
    Transpose a 4D array from (d0, d1, d2, d3) to (d0, d3, d1, d2) using Cython ijk implementation.

    Args:
        original: The input 4D array.
        transposed: Optional pre-allocated output array.

    Returns:
        The transposed array.
    """
    d0, d1, d2, d3 = original.shape
    if transposed is None:
        transposed = np.empty((d0, d3, d1, d2), original.dtype)
    transpose_0312_ijk_cython(original, transposed)
    return transposed


def transpose_0312_ikj_cython_wrapper(
    original: np.ndarray,
    transposed: np.ndarray | None = None,
) -> np.ndarray:
    """
    Transpose a 4D array from (d0, d1, d2, d3) to (d0, d3, d1, d2) using Cython ikj implementation.

    Args:
        original: The input 4D array.
        transposed: Optional pre-allocated output array.

    Returns:
        The transposed array.
    """
    d0, d1, d2, d3 = original.shape
    if transposed is None:
        transposed = np.empty((d0, d3, d1, d2), original.dtype)
    transpose_0312_ikj_cython(original, transposed)
    return transposed


best_transpose_0312: Callable[[np.ndarray, np.ndarray | None], np.ndarray] = BestOf(
    name="Transpose 0312 methods",
    alternatives=[
        ("numpy", transpose_0312_numpy),
        ("ijk_cyt", transpose_0312_ijk_cython_wrapper),
        ("ikj_cyt", transpose_0312_ikj_cython_wrapper),
    ],
    get_problem_size=lambda *args: args[0].shape,
)
