import numpy as _np
type _npDT = _np.int8 | _np.float32 | _np.float64
type _npDT_4Dims[T] = _np.ndarray[tuple[int, int, int, int], T]
type _npDT_3Dims[T] = _np.ndarray[tuple[int, int, int], T]
type _npDT_2Dims[T] = _np.ndarray[tuple[int, int], T]
type _npDT_1Dims[T] = _np.ndarray[tuple[int], T]


def compute_dense_acc_cython(residuals: _npDT_2Dims[_np.float32],
                             dw: _npDT_2Dims[_np.float32],
                             learning_rate: float) -> _npDT_2Dims[_np.float32]:
    ...


def intersect_2d_indexes_cython(local_rows: _npDT_1Dims[_np.int32],
                                local_cols: _npDT_1Dims[_np.int32],
                                global_rows: _npDT_1Dims[_np.int32],
                                global_cols: _npDT_1Dims[_np.int32]) -> tuple[_npDT_1Dims[_np.int32], _npDT_1Dims[_np.int32]]:
    ...


def reset_residuals_cython(acc: _npDT_2Dims[_np.float32],
                           rows: _npDT_1Dims[_np.int32],
                           cols: _npDT_1Dims[_np.int32]) -> _npDT_2Dims[_np.float32]:
    ...


def update_dense_weights_cython(w: _npDT_2Dims[_np.float32],
                                u: _npDT_2Dims[_np.float32]) -> _npDT_2Dims[_np.float32]:
    ...


def update_sparsed_weights_cython(w: _npDT_2Dims[_np.float32],
                                  grads_to_update: _npDT_1Dims[_np.float32],
                                  rows_to_update: _npDT_1Dims[_np.int32],
                                  cols_to_update: _npDT_1Dims[_np.int32]) -> _npDT_2Dims[_np.float32]:
    ...


def update_sparsed_weights_mv_cython(w: _npDT_2Dims[_np.float32],
                                     grads_to_update: _npDT_1Dims[_np.float32],
                                     rows_to_update: _npDT_1Dims[_np.int32],
                                     cols_to_update: _npDT_1Dims[_np.int32],
                                     velocity: _npDT_2Dims[_np.float32],
                                     momentum: float) -> tuple[_npDT_2Dims[_np.float32], _npDT_2Dims[_np.float32]]:
    ...
