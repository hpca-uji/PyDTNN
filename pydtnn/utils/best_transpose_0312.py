import numpy as np
from pydtnn.backends.cpu.utils.transpose_cython import transpose_0312_ijk_cython, transpose_0312_ikj_cython

from pydtnn.utils.best_of import BestOf
from typing import Callable


def transpose_0312_numpy(original: np.ndarray, 
                         transposed: np.ndarray | None = None  # type: ignore
                         ) -> np.ndarray:
    d0, d1, d2, d3 = original.shape
    if transposed is None:
        transposed: np.ndarray = np.empty((d0, d3, d1, d2), original.dtype, order="C")
    transposed[...] = original.transpose((0, 3, 1, 2))
    return transposed


def transpose_0312_ijk_cython_wrapper(original: np.ndarray, 
                                      transposed: np.ndarray | None = None  # type: ignore
                                      ) -> np.ndarray:
    d0, d1, d2, d3 = original.shape
    if transposed is None:
        transposed: np.ndarray = np.empty((d0, d3, d1, d2), original.dtype, order="C")
    transpose_0312_ijk_cython(original, transposed)
    return transposed


def transpose_0312_ikj_cython_wrapper(original: np.ndarray, 
                                      transposed: np.ndarray | None = None  # type: ignore
                                      ) -> np.ndarray:
    d0, d1, d2, d3 = original.shape
    if transposed is None:
        transposed: np.ndarray = np.empty((d0, d3, d1, d2), original.dtype, order="C")
    transpose_0312_ikj_cython(original, transposed)
    return transposed


best_transpose_0312: Callable[[np.ndarray, np.ndarray | None], np.ndarray] = BestOf(
    name="Transpose 0312 methods",
    alternatives=[
        ("ijk_cyt", transpose_0312_ijk_cython_wrapper),
        ("ikj_cyt", transpose_0312_ikj_cython_wrapper),
        ("numpy", transpose_0312_numpy),
    ],
    get_problem_size=lambda *args: args[0].shape,
)
