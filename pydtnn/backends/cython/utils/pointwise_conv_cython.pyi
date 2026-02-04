import numpy as _np
type _npDT = _np.int8 | _np.float32 | _np.float64
type _npDT_4Dims[T] = _np.ndarray[tuple[int, int, int, int], T]
type _npDT_3Dims[T] = _np.ndarray[tuple[int, int, int], T]
type _npDT_2Dims[T] = _np.ndarray[tuple[int, int], T]
type _npDT_1Dims[T] = _np.ndarray[tuple[int], T]


def pointwise_conv_cython[T: _npDT](x: _npDT_4Dims[T], k: _npDT_2Dims[T], out: _npDT_4Dims[T]) -> None:
    """
    Args:
        x (npDT_4Dims): 4-dimensinal array where the input data is stored.
        k (npDT_2Dims): 2-dimensinal array where the kernel is stored.
        out (npDT_4Dims): 4-dimensinal array where the output is stored.
    Returns:
        Nothing. The output is stored in \"out\".
    """
