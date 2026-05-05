import numpy as _np

from pydtnn.backends.cython.utils.base import _npDT, _npDT_1Dims, _npDT_2Dims, _npDT_3Dims, _npDT_4Dims

def argmax_cython[T: _npDT](x: _npDT_2Dims[T],
                            maxv: _npDT_1Dims[T],
                            amax: _npDT_1Dims[_np.int32],
                            rng: _npDT_1Dims[_np.int32],
                            axis: int = 0) -> tuple[_npDT_1Dims[T], _npDT_1Dims[T]]:
    """
    Args:
        x (npDT_2Dims): A view 2 dimensional inptu's ndarray.
        maxv (npDT_1Dims): A view to a ndarray of one of the npDT's types where the max values' will be stored.
        amax (np.ndarray[tuple[int], np.int32]): view to a ndarray of type np.int32 where the arg max values' will be stored.
        rng (np.ndarray[tuple[int], np.int32]): view to a ndarray of type np.int32 where some outputs will be stored.
        axis (int): The axis where the argmax will be performed. Can be 0 or 1. Default: 0.

    Returns:
        Explicit: tuple[np.ndarray, np.ndarray]: a tuple formed by: [T: npDT](amax, rng) if axis is 0, or [T: npDT](rng, amax) if not.

        Implicit: maxv.
    """
