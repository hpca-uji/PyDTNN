import numpy as _np
type _npDT = _np.int8 | _np.float32 | _np.float64
type _npDT_4Dims[T] = _np.ndarray[tuple[int, int, int, int], T]
type _npDT_3Dims[T] = _np.ndarray[tuple[int, int, int], T]
type _npDT_2Dims[T] = _np.ndarray[tuple[int, int], T]
type _npDT_1Dims[T] = _np.ndarray[tuple[int], T]


def adaptive_avg_pooling_fwd_nhwc_cython[T: _npDT](x: _npDT_4Dims[T], pooled_x: _npDT_4Dims[T]) -> None:
    """
    Args:
        x (npDT_4Dims): data input.
        pooled_x (npDT_4Dims): ndarray where the output will be stored.
    Returns:
        Nothing; the return is stored in "dx".
    """


def adaptive_avg_pooling_bwd_nhwc_cython[T: _npDT](dy: _npDT_4Dims[T], dx: _npDT_4Dims[T]) -> None:
    """
    Args:
        dy (npDT_4Dims): data input.
        dx (npDT_4Dims): ndarray where the output will be stored.
    Returns:
        Nothing; the return is stored in "dx".
    """
