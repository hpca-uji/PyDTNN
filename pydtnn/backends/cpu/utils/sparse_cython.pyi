import numpy as _np
type _npDT = _np.int8 | _np.float32 | _np.float64
type _npDT_4Dims[T] = _np.ndarray[tuple[int, int, int, int], T]
type _npDT_3Dims[T] = _np.ndarray[tuple[int, int, int], T]
type _npDT_2Dims[T] = _np.ndarray[tuple[int, int], T]
type _npDT_1Dims[T] = _np.ndarray[tuple[int], T]


def summ_coo_cython(self_data: _npDT_1Dims[_np.float32],
                    self_rows: _npDT_1Dims[_np.int32],
                    self_cols: _npDT_1Dims[_np.int32],
                    other_data: _npDT_1Dims[_np.float32],
                    other_rows: _npDT_1Dims[_np.int32],
                    other_cols: _npDT_1Dims[_np.int32]) -> tuple[_npDT_1Dims[_np.float32], _npDT_1Dims[_np.int32], _npDT_1Dims[_np.int32]]:
    ...


def top_threshold_selection_dense_cython(matrix: _npDT_2Dims[_np.float32],
                                         threshold: float) -> tuple[_npDT_1Dims[_np.float32], _npDT_1Dims[_np.int32], _npDT_1Dims[_np.int32]]:
    ...


def top_threshold_selection_coo_cython(values: _npDT_1Dims[_np.float32],
                                       rows: _npDT_1Dims[_np.int32],
                                       cols: _npDT_1Dims[_np.int32],
                                       threshold: float) -> tuple[_npDT_1Dims[_np.float32], _npDT_1Dims[_np.int32], _npDT_1Dims[_np.int32]]:
    ...
