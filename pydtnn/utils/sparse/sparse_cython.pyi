import numpy as _np
from pydtnn.backends.cython.utils.base import _npDT, _npDT_4Dims, _npDT_3Dims, _npDT_2Dims, _npDT_1Dims


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
