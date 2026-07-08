"""
Sparse matrix utilities for the PyDTNN framework.

This module provides the SparseMatrixCOO class, which implements a Coordinate (COO)
sparse matrix format optimized for performance using Cython-backed operations.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from pydtnn.backends.cython.utils.base import _npDT  # type: ignore

from pydtnn.utils.sparse.sparse_cython import (summ_coo_cython, top_threshold_selection_coo_cython,
                                               top_threshold_selection_dense_cython)

__all__ = ("SparseMatrixCOO",)

logger = logging.getLogger(__name__)


type DataType[T: _npDT] = np.ndarray[tuple[int], np.dtype[T]]
type IndexType = np.ndarray[tuple[int, int], np.dtype[np.int32]]
type RowType = np.ndarray[tuple[int], np.dtype[np.int32]]
type ColType = RowType

class SparseMatrixCOO[T: _npDT]:  # noqa: D101 (generics not detected)
    """Represents a sparse matrix in COO format.

    This format stores the matrix using three arrays:
        - data: the nonzero values.
        - row: the row indices corresponding to each value.
        - col: the column indices corresponding to each value.

    The matrix is assumed to be in canonical format: indices sorted by row and then by column,
    and no duplicate entries are present.
    This class is not designed to store explict zeros so, len(self.data) should always be equal to nnz.
    """

    def __init__(
        self,
        data: DataType[T],
        indexes: IndexType,
        shape: tuple,
        has_canonical_format: bool,
    ) -> None:
        """Primary initializer for SparseMatrixCOO.

        Parameters:
            data (np.ndarray[tuple[int], np.dtype[T]]): Array with the nonzero values.
            indexes  (np.np.ndarray[tuple[int, int], np.dtype[np.int32]]): Array with the row and column indices.
            shape (tuple): Shape of the original matrix.
            has_canonical_format (bool): Whether the input arrays are already sorted.
        """

        if len(data) != len(indexes):
            raise AssertionError("Data and indexes arrays must have the same shape")

        if has_canonical_format:
            self.data: DataType[T] = data
            self.indexes: IndexType = indexes
            self.shape: tuple = shape
            self.nnz: int = len(self.data)
            self.has_canonical_format: bool = True
            assert self._has_canonical_format()

        else:
            # TODO: order arrays in canonical format
            raise NotImplementedError(
                "Not yet implemented constructor with unordered rows and cols"
            )

    @property
    def row(self) -> RowType:
        return self.indexes[:, :0]
    
    @property
    def col(self) -> ColType:
        return self.indexes[:, :1]

    @staticmethod
    def to_indexes(row: RowType, col: ColType) -> IndexType:
        return np.array(list(zip(row, col)), dtype=np.int32)

    @classmethod
    def from_dense(cls, dense_array: np.ndarray) -> SparseMatrixCOO[T]:
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
        indexes = cls.to_indexes(*map(np.ndarray.tolist, _indexes))
        return cls(data, indexes, dense_array.shape, has_canonical_format=True)

    @classmethod
    def from_dense_top_selection(
        cls, dense_array: np.ndarray, threshold: float
    ) -> SparseMatrixCOO[T]:
        """Constructor from a dense array considering only elements greater than or equal to the threshold.

        Parameters:
            dense_array (np.ndarray): A 2D dense matrix.
            threshold (float): Threshold for including an element.

        Returns:
            SparseMatrixCOO: The sparse matrix in COO format, containing only significant elements.
        """

        if len(dense_array.shape) != 2:
            raise AssertionError("Dense array must be 2D.")

        # topk_row, topk_col = np.where(np.abs(dense_array) >= threshold)
        # topk = dense_array[topk_row, topk_col]
        topk, topk_row, topk_col = top_threshold_selection_dense_cython(dense_array, threshold)
        indexes = cls.to_indexes(topk_row, topk_col)
        return cls(topk, indexes, dense_array.shape, has_canonical_format=True)

    def top_selection(
        self, threshold: float, inplace: bool | None = True
    ) -> SparseMatrixCOO[T] | None:
        """
        Performs top threshold selection on sparse array

        Parameters:
            threshold (float): Threshold for including an element.
            inplace (bool, optional): Whether to modify the current instance.

        Returns:
            topk (SparseMatrixCOO[T]): if inplace == False, or void (None): if inplace == True
        """

        topk, topk_row, topk_col = top_threshold_selection_coo_cython(
            self.data, self.row, self.col, threshold
        )
        indexes = self.to_indexes(topk_row, topk_col)

        if inplace:
            self.data = topk
            self.indexes = indexes
            self.nnz = len(self.data)
            # self.shape remains equal
            # self.has_canonical_format remains equal
        else:
            return SparseMatrixCOO[T](
                topk, indexes, self.shape, self.has_canonical_format
            )

    def get_indexes(self) -> tuple[RowType, ColType]:
        """
        Returns the row and col indices.

        Returns:
            tuple: (row, col) arrays.
        """
        return self.row, self.col

    def get_triplet(self) -> tuple[DataType[T], RowType, ColType]:
        """
        Returns the data, row, col triplet.

        Returns:
            tuple: (data, row, col) arrays.
        """
        return self.data, *self.get_indexes()

    def slice(
        self, row_start: int, row_end: int, reset_indexes: bool | None = False
    ) -> SparseMatrixCOO[T]:
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
        start_index = np.searchsorted(self.row, row_start, side="left")
        ending_index = np.searchsorted(self.row, row_end, side="left")

        sliced_data = self.data[start_index:ending_index]
        sliced_row = self.row[start_index:ending_index]
        sliced_col = self.col[start_index:ending_index]
        if reset_indexes:
            sliced_row -= row_start

        sliced_indexes = self.to_indexes(sliced_row, sliced_col)

        return SparseMatrixCOO[T](
            sliced_data, sliced_indexes, self.shape, self.has_canonical_format
        )

    def to_dense(self) -> np.ndarray[tuple[int, ...], np.dtype[np.float32]]:
        """
        Convert to dense np.array.

        Returns:
            np.array: A dense matrix representation.
        """

        logger.warning(
            "This function ('to_sparse') should be used only in case of debugging for performance"
            " reasons."
        )

        dense_matrix = np.zeros(self.shape, dtype=np.float32)
        dense_matrix[self.row, self.col] = self.data
        return dense_matrix

    def __add__(self, other: SparseMatrixCOO[T]) -> SparseMatrixCOO[T]:
        """
        Adds two SparseMatrixCOO matrices that are in canonical format.

        Parameters:
            other (SparseMatrixCOO[T]): Another SparseMatrixCOO instance.

        Returns:
            SparseMatrixCOO: A new instance representing the sum of both matrices.
        """

        if not isinstance(other, SparseMatrixCOO):
            raise AssertionError("Operand must be a SparseMatrixCOO instance.")
        if self.shape != other.shape:
            raise AssertionError("Matrices must have the same shape.")
        if not self.has_canonical_format or not other.has_canonical_format:
            raise AssertionError("Both matrices must be in canonical format.")

        summ_val, summ_row, summ_col = summ_coo_cython(
            self.data, self.row, self.col, other.data, other.row, other.col
        )
        return SparseMatrixCOO[T](
            summ_val, self.to_indexes(summ_row, summ_col), self.shape, has_canonical_format=True
        )

    def __radd__(self, other: int | SparseMatrixCOO[T]) -> SparseMatrixCOO[T]:
        """
        Implements right-hand addition to support the built-in sum() function.

        This method allows an instance of this class to be used with sum() by handling the
        case where the left operand is 0. If 'other' is 0, it returns the instance itself;
        otherwise, it delegates the operation to the __add__ method.

        Parameters:
            other (int or SparseMatrixCOO[T]): The left-hand operand.

        Returns:
            SparseMatrixCOO: The sum of self and other.
        """
        if other == 0:
            return self
        else:
            assert not isinstance(other, int)
            return self.__add__(other)

    def _has_canonical_format(self) -> bool:
        """Check if SparseMatrixCOO follows canonical format.

        Canonical format:
            - Indexes are sorted by row and then by column
            - There are no duplicate entries
            - There may have explicit zero elements

        This function is computationally expensive and therefore should only be used for developing/debugging purposes.
        This function should only be used in developement to assert that sparse matrices have canonical format.

        Returns:
            bool: True if indexes are in canonical format, False if not.
        """

        logger.warning(
            "This function ('has_canonical_format') should be used only in case of debugging for"
            " performance reasons."
        )

        if self.nnz == 0:
            return True

        if not np.all(self.row[:-1] <= self.row[1:]):
            return False

        for i in range(self.nnz - 1):
            if self.row[i] == self.row[i + 1] and self.col[i] >= self.col[i + 1]:
                return False
        return True
