"""
Sparse matrix utilities for the PyDTNN framework.

This module provides the SparseMatrixFlat class, which implements a Flatten Coordinate (FCOO)
sparse matrix format optimized for performance using Cython-backed operations.
"""

from __future__ import annotations

import copy
import logging
import math
from typing import TYPE_CHECKING, Any, Self

import numpy as np

if TYPE_CHECKING:
    from pydtnn.backends.cython.utils.base import _npDT  # type: ignore

from pydtnn.utils.constants import ArrayShape
from pydtnn.utils.sparse.sparse_cython import (summ_coo_cython, top_threshold_selection_coo_cython,
                                               top_threshold_selection_dense_cython)

__all__ = ("SparseMatrixFlat",)

logger = logging.getLogger(__name__)


type DataType[T: _npDT] = np.ndarray[tuple[int], np.dtype[T]]
type IndexType = np.ndarray[tuple[int], np.dtype[np.int32]]
type RowType = np.ndarray[tuple[int], np.dtype[np.int32]]
type ColType = RowType


class SparseMatrixFlat[T: _npDT]:  # noqa: D101 (generics not detected)
    """Represents a sparse matrix in Flatten COO format.

    This format stores the matrix using three arrays:
        - data: the nonzero values.
        - row: the row indices corresponding to each value.
        - col: the column indices corresponding to each value.

    and no duplicate entries are present.
    This class is not designed to store explicit zeros so, len(self.data) should always be equal to number_non_zeros.
    """

    def __init__(self, data: DataType[T], indexes: IndexType, shape: ArrayShape) -> None:
        """Primary initializer for SparseMatrixCOO.

        Parameters:
            data (np.ndarray[tuple[int], np.dtype[T]]): Array with the nonzero values.
            indexes (np.np.ndarray[tuple[int], np.dtype[np.int32]]): Array with the values' position
              as if the array is flattened.
            shape (tuple): Shape of the original matrix.
        """

        if len(data) != len(indexes):
            raise AssertionError("Data and indexes arrays must have the same shape")

        self.data: DataType[T] = data
        self.indexes: IndexType = indexes
        self.shape: ArrayShape = shape
        assert self._has_canonical_format()

    @property
    def number_non_zeros(self) -> int:
        return len(self.data)

    @property
    def row(self) -> RowType:
        """Row indexes"""
        return self.indexes[:, :0]

    @property
    def col(self) -> ColType:
        """Column indexes"""
        return self.indexes[:, :1]

    @staticmethod
    def to_indexes(row: RowType, col: ColType) -> IndexType:
        """Merge rows and columns to an index array"""
        return np.array(list(zip(row, col)), dtype=np.int32)

    @classmethod
    def from_unsorted_indexes(
        cls, data: DataType, indexes: IndexType, shape: ArrayShape
    ) -> SparseMatrixFlat[T]:
        """Constructs to create a SparseMatrixCOO from a unsorted indexes array.

        Parameters:
            data (np.ndarray[tuple[int], np.dtype[T]]): Array with the nonzero values.
            indexes (np.np.ndarray[tuple[int], np.dtype[np.int32]]): Unsorted array with the values' position
                as if the array is flattened.
            shape (tuple): Shape of the original matrix.

        Returns:
            SparseMatrixCOO: The sparse matrix in COO format
        """

        indexes_and_data = list(zip(data, indexes))
        indexes_and_data.sort(key=lambda x: x[0])
        indexes, data = zip(*indexes_and_data)  # type: ignore (It's the right data type)

        return cls(data, indexes, shape)

    @classmethod
    def from_dense(cls, dense_array: np.ndarray) -> SparseMatrixFlat[T]:
        """Constructs to create a SparseMatrixCOO from a dense array.

        Only stores non-zero values!

        Parameters:
            dense_array (np.ndarray): A 2D dense matrix.

        Returns:
            SparseMatrixCOO: The sparse matrix in COO format
        """

        if len(dense_array.shape) != 2:
            raise AssertionError("Dense array must be 2D.")

        logger.warning(
            "From dense constructor should be used only in case of debugging for performance"
            " reasons."
        )

        _indexes = np.where(dense_array != 0)
        data = dense_array[_indexes]
        indexes = cls.to_indexes(*_indexes)  # type: ignore
        return cls(data, indexes, dense_array.shape)

    @classmethod
    def from_dense_top_selection(
        cls, dense_array: np.ndarray, threshold: float
    ) -> SparseMatrixFlat[T]:
        """Constructor from a dense array considering only elements greater than or equal to the threshold.

        Parameters:
            dense_array (np.ndarray): A 2D dense matrix.
            threshold (float): Threshold for including an element.

        Returns:
            SparseMatrixCOO: The sparse matrix in COO format, containing only significant elements.
        """

        if len(dense_array.shape) != 2:
            raise AssertionError("Dense array must be 2D.")
        shape = dense_array.size
        top_values = np.zeros(shape, dtype=dense_array.dtype)
        top_indices = np.zeros(shape, dtype=np.int32)

        topk, topk_indexes = top_threshold_selection_dense_cython(
            dense_array, threshold, top_values, top_indices
        )
        return cls(topk, topk_indexes, dense_array.shape)

    @staticmethod
    def intersection_indexes(o1: np.ndarray, o2: np.ndarray) -> np.ndarray:
        """Returns the intersection of two SparseMatrixCOO's indexes."""
        return np.intersect1d(o1, o2)

    def __copy__(self) -> Self:
        """Return copy of SparseMatrixCOO"""
        return self.__class__(self.data.copy(), self.indexes.copy(), self.shape)

    def copy(self) -> Self:
        """Return copy of SparseMatrixCOO"""
        return copy.copy(self)

    def __and__(self, other: Any) -> SparseMatrixFlat[T]:
        """Returns a new SparseMatrixCOO with the intersection of self and other"""
        if not isinstance(other, SparseMatrixFlat):
            raise TypeError("Operand must be a SparseMatrixCOO instance.")
        if self.shape != other.shape:
            raise ValueError("Matrices must have the same shape.")
        if not self._has_canonical_format() or not other._has_canonical_format():
            raise ValueError("Both matrices must be sorted.")
        indexes = self.intersection_indexes(self.indexes, other.indexes)
        return SparseMatrixFlat[T](data=self.data[indexes], indexes=indexes, shape=self.shape)

    def __iand__(self, other: Any) -> Self:
        """Returns a new SparseMatrixCOO with the intersection of self and other"""
        if not isinstance(other, SparseMatrixFlat):
            raise TypeError("Operand must be a SparseMatrixCOO instance.")
        if self.shape != other.shape:
            raise ValueError("Matrices must have the same shape.")
        if not self._has_canonical_format() or not other._has_canonical_format():
            raise ValueError("Both matrices must be sorted.")
        self.indexes = self.intersection_indexes(self.indexes, other.indexes)
        self.data = self.data[self.indexes]
        return self

    def intersection(self, other: SparseMatrixFlat, inplace: bool = False) -> SparseMatrixFlat[T]:
        """Returns a new SparseMatrixCOO with the intersection of self and other, isolated or inplace"""
        if inplace:
            self &= other
        else:
            self = self & other
        return self

    def threshold_selection(
        self, threshold: float, inplace: bool | None = True
    ) -> SparseMatrixFlat[T] | None:
        """
        Performs top threshold selection on sparse array

        Parameters:
            threshold (float): Threshold for including an element.
            inplace (bool, optional): Whether to modify the current instance.

        Returns:
            topk (SparseMatrixCOO[T]): if inplace == False, or void (None): if inplace == True
        """

        shape = math.prod(self.shape)
        top_values = np.zeros((shape,), dtype=self.data.dtype)
        top_indices = np.zeros((shape,), dtype=np.int32)

        topk, topk_indixes = top_threshold_selection_coo_cython(
            self.data, self.indexes, threshold, top_values, top_indices
        )

        if inplace:
            self.data = topk
            self.indexes = topk_indixes
            # self.shape remains equal
        else:
            return SparseMatrixFlat[T](topk, self.indexes, self.shape)

    def get_data_and_indexes(self) -> tuple[DataType[T], IndexType]:
        """
        Returns the data and the indexes.

        Returns:
            tuple: (data, indexes) arrays.
        """
        return self.data, self.indexes

    def slice_selection(
        self, row_start: int, row_end: int, reset_indexes: bool | None = False
    ) -> SparseMatrixFlat[T]:
        """
        Perform a slice by row of the sparse matrix.

        Parameters:
            row_start (int): The starting row index (inclusive) of the slice.
            row_end (int): The ending row index (exclusive) of the slice.
            reset_indexes (bool, optional): If True, resets the row indices of the
                                            sliced matrix so that `row_start` maps to zero.
                                            Defaults to False.
        Returns:
            SparseMatrixCOO: A row-sliced sparse matrix of self.
        """
        # Converting from matrix/tensor row to a flattened matrix/tensor position:
        rows = self.shape[0]
        not_rows = math.prod(self.shape) // rows
        flattened_pos_start = row_start * not_rows
        flattened_pos_end = row_end * not_rows

        start_index = np.searchsorted(self.indexes, flattened_pos_start, side="left")
        ending_index = np.searchsorted(self.indexes, flattened_pos_end, side="left")

        sliced_data = self.data[start_index:ending_index]
        sliced_indexes = self.indexes[start_index:ending_index]

        # TODO: Check this works as intended:
        if reset_indexes:
            sliced_indexes -= flattened_pos_start

        return SparseMatrixFlat[T](sliced_data, sliced_indexes, self.shape)

    def to_dense(self) -> np.ndarray[tuple[int, ...], np.dtype[T]]:
        """
        Convert to dense np.array.

        Returns:
            np.array: A dense matrix representation.
        """

        logger.warning(
            "This function ('to_sparse') should be used only in case of debugging for performance"
            " reasons."
        )

        shape = math.prod(self.shape)
        dense_matrix = np.zeros((shape,), dtype=self.data.dtype)
        dense_matrix[self.indexes] = self.data
        return dense_matrix.reshape(self.shape)

    def __iadd__(self, other: SparseMatrixFlat[T]) -> Self:
        """
        Adds two SparseMatrixCOO matrices that are sorted.

        Parameters:
            other (SparseMatrixCOO[T]): Another SparseMatrixCOO instance.

        Returns:
            SparseMatrixCOO: A new instance representing the sum of both matrices.
        """

        if other == 0:
            return self
        if not isinstance(other, SparseMatrixFlat):
            raise TypeError("Operand must be a SparseMatrixCOO instance.")
        if self.shape != other.shape:
            raise ValueError("Matrices must have the same shape.")
        if not self._has_canonical_format() or not other._has_canonical_format():
            raise ValueError("Both matrices must be sorted.")

        max_size = self.number_non_zeros + other.number_non_zeros
        summ_val = np.zeros(max_size, dtype=self.data.dtype)
        summ_indices = np.zeros(max_size, dtype=np.int32)
        self.data, self.indexes = summ_coo_cython(
            self.data, self.indexes, other.data, other.indexes, summ_val, summ_indices
        )
        return self

    def __add__(self, other: SparseMatrixFlat[T]) -> SparseMatrixFlat[T]:
        """
        Adds two SparseMatrixCOO matrices that are sorted.

        Parameters:
            other (SparseMatrixCOO[T]): Another SparseMatrixCOO instance.

        Returns:
            SparseMatrixCOO: A new instance representing the sum of both matrices.
        """

        if other == 0:
            return self
        if not isinstance(other, SparseMatrixFlat):
            raise TypeError("Operand must be a SparseMatrixCOO instance.")
        if self.shape != other.shape:
            raise ValueError("Matrices must have the same shape.")
        if not self._has_canonical_format() or not other._has_canonical_format():
            raise ValueError("Both matrices must be sorted.")

        max_size = self.number_non_zeros + other.number_non_zeros
        summ_val = np.zeros(max_size, dtype=self.data.dtype)
        summ_indices = np.zeros(max_size, dtype=np.int32)
        summ_val, summ_indices = summ_coo_cython(
            self.data, self.indexes, other.data, other.indexes, summ_val, summ_indices
        )
        return SparseMatrixFlat[T](summ_val, summ_indices, self.shape)

    def __radd__(self, other: SparseMatrixFlat[T]) -> SparseMatrixFlat[T]:
        """Reversed add"""
        return self + other

    def _has_canonical_format(self) -> bool:
        """Check if SparseMatrixCOO follows canonical format.

        Canonical format:
            - Indexes are sorted by row and then by column
            - There are no duplicate entries
            - There may have explicit zero elements

        This function is computationally expensive and therefore should only be used for developing/debugging purposes.
        This function should only be used in developement to assert that sparse matrices have canonical format.

        Returns:
            bool: True if indexes are sorted, False if not.
        """

        logger.warning(
            "This function ('has_canonical_format') should be used only in case of debugging for"
            " performance reasons."
        )

        return (self.number_non_zeros == 0) or bool(np.all(self.indexes[:-1] < self.indexes[1:]))
