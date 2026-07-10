"""Cython-accelerated utility functions for sparse matrix operations."""

import numpy as _np

from pydtnn.backends.cython.utils.base import _npDT, _npDT_1Dims, _npDT_2Dims

def summ_coo_cython[T: _npDT](
    self_data: _npDT_1Dims[T],
    self_indices: _npDT_1Dims[_np.int32],
    other_data: _npDT_1Dims[T],
    other_indices: _npDT_1Dims[_np.int32],
    summ_data: _npDT_1Dims[T],
    summ_indices: _npDT_1Dims[_np.int32],
) -> tuple[_npDT_1Dims[T], _npDT_1Dims[_np.int32]]:
    """Sum two sparse matrices in COO format.
    Args:
        self_data (_npDT_1Dims[T]): array where the the base matrix's data (self) is stored.
        other_data (_npDT_1Dims[T]): array where the other matrix's data is stored.
        self_indices (_npDT_1Dims[_np.int32]): array where the the base matrix's indexes are stored.
        other_indices (_npDT_1Dims[_np.int32]): array where the other matrix's indexes are stored.
        summ_data (_npDT_1Dims[T]): array where the result is stored (It's expected that len(summ) = len(self) + len(other)).
        summ_indices (_npDT_1Dims[_np.int32]): array where the result indexes are stored (It's expected that len(summ) = len(self) + len(other)).
    Output:
        tuple[np.ndarrays[_npDT], np.ndarrays[_npDT]]
        - It returns a tuple with a slice of summ_data and a slice of summ_indices.
    """

    ...

def top_threshold_selection_dense_cython[T: _npDT](
    matrix: _npDT_2Dims[T],
    threshold: float,
    top_values: _npDT_1Dims[T],
    top_indices: _npDT_1Dims[_np.int32],
) -> tuple[_npDT_1Dims[T], _npDT_1Dims[_np.int32]]:
    """Select elements from a dense matrix above a threshold and return in COO format.
    Args:
        matrix (_npDT_2Dims[T]): dense matrix.
        threshold (float): The value that sets what elements that will be stored (values below the threshold are discarted).
        top_values (_npDT_1Dims[T]): array where the result is stored [It's expected that len(top_values) = len(matrix)].
        top_indices (_npDT_1Dims[_np.int32]): array where the result indexes are stored [It's expected that len(top_indices) = len(matrix)].
    Output:
        tuple[np.ndarrays[_npDT], np.ndarrays[_npDT]]
        - It returns a tuple with a slice of top_values and a slice of top_indices.
    """
    ...

def top_threshold_selection_coo_cython[T: _npDT](
    values: _npDT_1Dims[T],
    indices: _npDT_1Dims[_np.int32],
    threshold: float,
    top_values: _npDT_1Dims[T],
    top_indices: _npDT_1Dims[_np.int32],
) -> tuple[_npDT_1Dims[T], _npDT_1Dims[_np.int32]]:
    """Filter elements of a COO sparse matrix based on a threshold.
    Args:
        value (_npDT_1Dims[T]): sparse matrix.
        indices (_npDT_1Dims[T]): sparse matrix's indices.
        threshold (float): The value that sets what elements that will be stored (values below the threshold are discarted).
        top_values (_npDT_1Dims[T]): array where the result is stored [It's expected that len(top_values) = sparse's matrix shape number of elements].
        top_indices (_npDT_1Dims[_np.int32]): array where the result indexes are stored [It's expected that len(top_indices) = sparse's matrix shape number of elements].
    Output:
        tuple[np.ndarrays[_npDT], np.ndarrays[_npDT]]
        - It returns a tuple with a slice of top_values and a slice of top_indices.
    """
    ...
