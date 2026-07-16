"""Cython-accelerated utility functions for OktopK operations in PyDTNN."""

import numpy as _np

from pydtnn.backends.cython.utils.base import _npDT, _npDT_1Dims, _npDT_2Dims

def compute_dense_acc_cython[T: _npDT](
    residuals: _npDT_2Dims[T], dw: _npDT_2Dims[T], acc: _npDT_2Dims[T], learning_rate: float
) -> None:
    """
    Compute dense accumulation of residuals scaled by learning rate.

    Args:
        residuals (npDT_2Dims):
        dw (npDT_2Dims):
        acc (npDT_2Dims):
        learning_rate (float):
    Returns:
        Nothing; the output is stored in "acc".
    """
    ...

def intersect_1d_indexes_cython(
    local_indexes: _npDT_1Dims[_np.int32],
    global_indexes: _npDT_1Dims[_np.int32],
    intersected_indexes: _npDT_1Dims[_np.int32],
) -> _npDT_1Dims[_np.int32]:
    """
    Find the intersection of two sets of 1D indices.
    Args:
        local_indexes (_npDT_1Dims[np.int32]):
        global_indexes (_npDT_1Dims[np.int32]):
        intersected_indexes (_npDT_1Dims[np.int32]):
    Returns:
        intersected_indexes (_npDT_1Dims[np.int32]):
    """
    ...

def reset_residuals_cython[T: _npDT](acc: _npDT_2Dims[T], indexes: _npDT_1Dims[_np.int32]) -> None:
    """
    Reset specific residual values to zero based on provided indices.
    Args:
        acc (npDT_2Dims): accuracy
        indexes (npDT_1Dims):
    Returns:
        Nothing; the output is stored in "acc".
    """
    ...

def update_dense_weights_cython[T: _npDT](w: _npDT_2Dims[T], nprocs: int, u: _npDT_2Dims[T]) -> None:
    """
    Update dense weights by adding the provided update matrix.
    Args:
        w (npDT_2Dims): weights
        u (npDT_2Dims):
    Returns:
        Nothing; the output is stored in "w".
    """
    ...

def update_sparsed_weights_cython[T: _npDT](
    w: _npDT_1Dims[T], nprocs: int, grads_to_update: _npDT_1Dims[T], indexes_to_update: _npDT_1Dims[_np.int32]
) -> None:
    """
    Update sparse weights using coordinate-based gradient updates.
    Args:
        w (_npDT_2Dims[T]):
        grads_to_update (_npDT_1Dims[T]):
        indexes_to_update (_npDT_1Dims[_np.int32]):
    Returns:
        Nothing; the output is stored in "w".
    """
    ...

def update_sparsed_weights_mv_cython[T: _npDT](
    w: _npDT_1Dims[T],
    nprocs: int,
    grads_to_update: _npDT_1Dims[T],
    indexes_to_update: _npDT_1Dims[_np.int32],
    velocity: _npDT_1Dims[T],
    momentum: float,
) -> None:
    """
    Update sparse weights and velocity using momentum-based optimization.

    Args:
        w (npDT[:,::1]): layer's weights.
        grads_to_update (npDT[:,::1]): gradients to update.
        indexes_to_update (np.int32_t[::1]):
        velocity (npDT[:,::1]):
        momentum (float):
    Returns:
        Nothing; the output is stored in "w" and in "velocity".
    """
    ...
