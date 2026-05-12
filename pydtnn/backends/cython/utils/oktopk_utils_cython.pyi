"""Cython-accelerated utility functions for OktopK operations in PyDTNN."""

import numpy as _np

from pydtnn.backends.cython.utils.base import _npDT_1Dims, _npDT_2Dims

def compute_dense_acc_cython(residuals: _npDT_2Dims[_np.float32], dw: _npDT_2Dims[_np.float32], learning_rate: float) -> _npDT_2Dims[_np.float32]:
    """Compute dense accumulation of residuals scaled by learning rate."""
    ...
def intersect_2d_indexes_cython(
    local_rows: _npDT_1Dims[_np.int32], local_cols: _npDT_1Dims[_np.int32], global_rows: _npDT_1Dims[_np.int32], global_cols: _npDT_1Dims[_np.int32]
) -> tuple[_npDT_1Dims[_np.int32], _npDT_1Dims[_np.int32]]:
    """Find the intersection of two sets of 2D indices."""
    ...
def reset_residuals_cython(acc: _npDT_2Dims[_np.float32], rows: _npDT_1Dims[_np.int32], cols: _npDT_1Dims[_np.int32]) -> _npDT_2Dims[_np.float32]:
    """Reset specific residual values to zero based on provided indices."""
    ...
def update_dense_weights_cython(w: _npDT_2Dims[_np.float32], u: _npDT_2Dims[_np.float32]) -> _npDT_2Dims[_np.float32]:
    """Update dense weights by adding the provided update matrix."""
    ...
def update_sparsed_weights_cython(
    w: _npDT_2Dims[_np.float32], grads_to_update: _npDT_1Dims[_np.float32], rows_to_update: _npDT_1Dims[_np.int32], cols_to_update: _npDT_1Dims[_np.int32]
) -> _npDT_2Dims[_np.float32]:
    """Update sparse weights using coordinate-based gradient updates."""
    ...
def update_sparsed_weights_mv_cython(
    w: _npDT_2Dims[_np.float32],
    grads_to_update: _npDT_1Dims[_np.float32],
    rows_to_update: _npDT_1Dims[_np.int32],
    cols_to_update: _npDT_1Dims[_np.int32],
    velocity: _npDT_2Dims[_np.float32],
    momentum: float,
) -> tuple[_npDT_2Dims[_np.float32], _npDT_2Dims[_np.float32]]:
    """Update sparse weights and velocity using momentum-based optimization."""
    ...