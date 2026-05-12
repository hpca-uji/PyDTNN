"""Cython-accelerated utility functions for sparse matrix operations."""

import numpy as _np

from pydtnn.backends.cython.utils.base import _npDT_1Dims, _npDT_2Dims

def summ_coo_cython(
    self_data: _npDT_1Dims[_np.float32],
    self_rows: _npDT_1Dims[_np.int32],
    self_cols: _npDT_1Dims[_np.int32],
    other_data: _npDT_1Dims[_np.float32],
    other_rows: _npDT_1Dims[_np.int32],
    other_cols: _npDT_1Dims[_np.int32],
) -> tuple[_npDT_1Dims[_np.float32], _npDT_1Dims[_np.int32], _npDT_1Dims[_np.int32]]:
    """Sum two sparse matrices in COO format."""
    ...

def top_threshold_selection_dense_cython(matrix: _npDT_2Dims[_np.float32], threshold: float) -> tuple[_npDT_1Dims[_np.float32], _npDT_1Dims[_np.int32], _npDT_1Dims[_np.int32]]:
    """Select elements from a dense matrix above a threshold and return in COO format."""
    ...

def top_threshold_selection_coo_cython(
    values: _npDT_1Dims[_np.float32], rows: _npDT_1Dims[_np.int32], cols: _npDT_1Dims[_np.int32], threshold: float
) -> tuple[_npDT_1Dims[_np.float32], _npDT_1Dims[_np.int32], _npDT_1Dims[_np.int32]]:
    """Filter elements of a COO sparse matrix based on a threshold."""
    ...
