import numpy as _np
type _npDT = _np.int8 | _np.float32 | _np.float64
type _npDT_4Dims[T] = _np.ndarray[tuple[int, int, int, int], T]
type _npDT_3Dims[T] = _np.ndarray[tuple[int, int, int], T]
type _npDT_2Dims[T] = _np.ndarray[tuple[int, int], T]
type _npDT_1Dims[T] = _np.ndarray[tuple[int], T]


def add_cython[T: _npDT](x: _npDT_2Dims[T], b: _npDT_1Dims[T]) -> None:
    """
    Args:
        x (npDT_2Dims): A contiguous memory view of the data. Since all the operations are made inplace, it's also where the output it's stored.
        b (npDT_1Dims): A contiguous memory view of the bias.

    Returns:
        Nothing. The output is stored in "x".
    """
