import numpy as _np
type _npDT = _np.int8 | _np.float32 | _np.float64
type _npDT_4Dims[T] = _np.ndarray[tuple[int, int, int, int], T]
type _npDT_3Dims[T] = _np.ndarray[tuple[int, int, int], T]
type _npDT_2Dims[T] = _np.ndarray[tuple[int, int], T]
type _npDT_1Dims[T] = _np.ndarray[tuple[int], T]


def log_fwd_cython[T:_npDT](x: _npDT_1Dims[T], y: _npDT_1Dims[T]) -> None:
    """
    Args:
        x (npDT_1Dims): 1-dimensional input's array.
        y (npDT_1Dims): 1-dimensional array where the ouput is stored
    Returns:
        Nothing. The output is stored in "y".
    """


def log_bwd_cython[T:_npDT](dy: _npDT_1Dims[T], dx: _npDT_1Dims[T]) -> None:
    """
    Args:
        dy (npDT_1Dims): 1-dimensional input's array.
        dx (npDT_1Dims): 1-dimensional array where the output will be stored.
    Returns:
        Nothing. The output is stored in "dx".
    """
