import numpy as np
from pydtnn.backends.cpu.utils.transpose_cython import transpose_0231_ijk_cython, transpose_0231_ikj_cython

from pydtnn.utils.best_of import BestOf
from typing import Callable


def transpose_0231_numpy(original: np.ndarray, 
                         transposed: np.ndarray | None = None # type: ignore
                        ) -> np.ndarray:  
    d0, d1, d2, d3 = original.shape
    if transposed is None:
        transposed: np.ndarray = np.empty((d0, d2, d3, d1), original.dtype, order="C")
    transposed[...] = original.transpose((0, 2, 3, 1))
    return transposed


def transpose_0231_ijk_cython_wrapper(original: np.ndarray, 
                                      transposed: np.ndarray | None = None # type: ignore
                                      ) -> np.ndarray:
    d0, d1, d2, d3 = original.shape
    if transposed is None:
        transposed: np.ndarray = np.empty((d0, d2, d3, d1), original.dtype, order="C")
    transpose_0231_ijk_cython(original, transposed)
    return transposed


def transpose_0231_ikj_cython_wrapper(original: np.ndarray, 
                                      transposed: np.ndarray | None = None # type: ignore
                                      ) -> np.ndarray:
    d0, d1, d2, d3 = original.shape
    if transposed is None:
        transposed: np.ndarray = np.empty((d0, d2, d3, d1), original.dtype, order="C")
    transpose_0231_ikj_cython(original, transposed)
    return transposed

# TODO: change typing "Callable, etc."
best_transpose_0231: Callable[[np.ndarray, np.ndarray | None], np.ndarray] = BestOf(
    name="Transpose 0231 methods",
    alternatives=[
        ("ikj_cyt", transpose_0231_ikj_cython_wrapper),
        ("ijk_cyt", transpose_0231_ijk_cython_wrapper),
        ("numpy", transpose_0231_numpy),
    ],
    get_problem_size=lambda *args: args[0].shape,
)
