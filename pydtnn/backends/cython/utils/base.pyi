import numpy as _np

type _npDT = _np.int8 | _np.float32 | _np.float64
type _npDT_4Dims[T: _np.number] = _np.ndarray[tuple[int, int, int, int], _np.dtype[T]]
type _npDT_3Dims[T: _np.number] = _np.ndarray[tuple[int, int, int], _np.dtype[T]]
type _npDT_2Dims[T: _np.number] = _np.ndarray[tuple[int, int], _np.dtype[T]]
type _npDT_1Dims[T: _np.number] = _np.ndarray[tuple[int], _np.dtype[T]]
