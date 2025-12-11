import numpy as np
from pydtnn.backends.cpu.utils.transpose_cython import transpose_1023_ijk_cython, transpose_1023_jik_cython

from pydtnn.utils.best_of import BestOf
from typing import Callable


def transpose_1023_numpy(original: np.ndarray, 
                         transposed: np.ndarray | None = None # type: ignore
                         ) -> np.ndarray:
    d0, d1, d2, d3 = original.shape
    if transposed is None:
        transposed: np.ndarray = np.empty((d1, d0, d2, d3), original.dtype, order="C")
    transposed[...] = original.transpose((1, 0, 2, 3))
    return transposed


def transpose_1023_ijk_cython_wrapper(original: np.ndarray, 
                                      transposed: np.ndarray | None = None # type: ignore
                                      ) -> np.ndarray:
    d0, d1, d2, d3 = original.shape
    if transposed is None:
        transposed: np.ndarray = np.empty((d1, d0, d2, d3), original.dtype, order="C")
    transpose_1023_ijk_cython(original, transposed)
    return transposed


def transpose_1023_jik_cython_wrapper(original: np.ndarray, 
                                      transposed: np.ndarray | None = None # type: ignore
                                      ) -> np.ndarray:
    d0, d1, d2, d3 = original.shape
    if transposed is None:
        transposed: np.ndarray = np.empty((d1, d0, d2, d3), original.dtype, order="C")
    transpose_1023_jik_cython(original, transposed)
    return transposed


best_transpose_1023: Callable[[np.ndarray, np.ndarray | None], np.ndarray] = BestOf(
    name="Transpose 1023 methods",
    alternatives=[
        ("ijk_cyt", transpose_1023_ijk_cython_wrapper),
        ("jik_cyt", transpose_1023_jik_cython_wrapper),
        ("numpy", transpose_1023_numpy),
    ],
    get_problem_size=lambda m: m.shape,
)
