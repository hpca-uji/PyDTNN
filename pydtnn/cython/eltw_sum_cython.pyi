import numpy as _np
type _npDT = _np.int8 | _np.float32 | _np.float64
type _npDT_4Dims[T] = _np.ndarray[tuple[int, int, int, int], T]
type _npDT_3Dims[T] = _np.ndarray[tuple[int, int, int], T]
type _npDT_2Dims[T] = _np.ndarray[tuple[int, int], T]
type _npDT_1Dims[T] = _np.ndarray[tuple[int], T]


def eltw_sum_cython[T: _npDT](x_acc: _npDT_1Dims[T], x: _npDT_1Dims[T]) -> None:
    """
    This function adds the values of "x_acc" and "x" and accumulate them in "x_acc".
    Args:
        x_acc (npDT_1Dims): The 1 dimensional where the accumulation will be stored.
        x (npDT_1Dims): The 1 dimensional array with the data to accumulate.
    Returns:
        Nothing. The output is stored in "x_acc".
    """
