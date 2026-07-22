"""
Sparse matrix utilities for the PyDTNN framework.

This module provides the SparseFlatArray class, which implements a Flatten Coordinate (FCOO).
"""

from __future__ import annotations

import copy
import math
from typing import Any

import numpy as np

from pydtnn.utils.constants import ArrayShape  # noqa: F401

__all__ = ("SparseFlatArray",)


type FlatArray[T: np.number] = np.ndarray[tuple[int], np.dtype[T]]


class SparseFlatArray[V: np.number, I: np.integer = np.int32, S: tuple = ArrayShape]:  # noqa: D101 (generics not detected)
    """Sparse flatten array"""

    def __init__(self, shape: S, indexes: FlatArray[I], values: FlatArray[V]) -> None:
        """Construct array"""
        self.shape = shape
        self.indexes = indexes
        self.values = values

        if len(self.indexes) != len(self.values):
            raise ValueError("Mismatch indexes and data array")
        elif indexes.min(initial=self.size) < 0 or indexes.max(initial=0) >= self.size:
            raise ValueError("Indexes out of range of shape")
        elif len(self.values) > self.size:
            raise ValueError("Too many values for shape")
        assert self.is_canonical(), "Non canonical representation"

    def __repr__(self) -> str:
        """Sparse flatten array representation"""
        return f"{self.__class__.__name__}(shape={self.shape}, indexes={self.indexes}, values={self.values})"

    @property
    def size(self) -> int:
        """Number of dense slots"""
        return math.prod(self.shape)

    @property
    def nnz(self) -> int:
        """Number of sparse slots"""
        return self.values.size

    @property
    def nbytes(self) -> int:
        """Number of sparse bytes"""
        return self.indexes.nbytes + self.values.nbytes

    @classmethod
    def from_dense[DV: np.number, DI: np.integer, DS: tuple](
        cls: type[SparseFlatArray[DV, DI, DS]],
        array: np.ndarray[DS, np.dtype[DV]],
        dtype: type[DI] = np.int32,
    ) -> SparseFlatArray[DV, DI, DS]:
        """Construct from dense"""
        shape = array.shape
        indexes = np.flatnonzero(array).astype(dtype)
        values = array.reshape(-1)[indexes]
        return cls(shape, indexes, values)

    def to_dense(self) -> np.ndarray[S, np.dtype[V]]:
        """Convert to dense"""
        ary = np.zeros(self.shape, dtype=self.values.dtype)
        ary[self.indexes] = self.values
        return ary

    def __copy__(self) -> SparseFlatArray[V, I, S]:  # noqa: E741
        """Shallow copy (maintain backing arrays)"""
        return self.__class__(self.shape, self.indexes, self.values)

    def __deepcopy__(self, memo: dict) -> SparseFlatArray[V, I, S]:  # noqa: E741
        """Deep copy (copy backing arrays)"""
        ary = memo[id(self)] = self.__copy__()
        ary.shape = copy.deepcopy(self.shape, memo)
        ary.indexes = copy.deepcopy(self.indexes, memo)
        ary.values = copy.deepcopy(self.values, memo)
        return ary

    def copy(self) -> SparseFlatArray[V, I, S]:  # noqa: E741
        """Copy (including backing arrays)"""
        return copy.deepcopy(self)

    def is_canonical(self) -> bool:
        """Check indexes are sorted and unique"""
        return bool(np.all(self.indexes[1:] > self.indexes[:-1]))

    def canonical(self) -> SparseFlatArray[V, I, S]:  # noqa: E741
        """Get canonical representation"""

        # order
        order = np.argsort(self.indexes)
        indexes = self.indexes[order]
        values = self.values[order]

        # unique
        indexes, idx = np.unique(indexes, return_index=True)
        values = values[idx]

        return self.__class__(self.shape, indexes, values)

    def __add__(self, other: Any) -> SparseFlatArray[V, I, S]:  # noqa: E741
        """Add two arrays"""
        if not isinstance(other, SparseFlatArray):
            raise TypeError("Operand must be a SparseFlatArray instance")
        elif self.shape != other.shape:
            raise ValueError("Array must have the same shape")

        # union
        indexes = np.union1d(self.indexes, other.indexes)
        values = np.zeros(indexes.size, dtype=self.values.dtype)

        # self
        insert = np.searchsorted(indexes, self.indexes)
        values[insert] += self.values

        # other
        insert = np.searchsorted(indexes, other.indexes)
        values[insert] += other.values

        return self.__class__(self.shape, indexes, values)

    def __getitem__(self, key: Any) -> SparseFlatArray[V, I, S]:  # noqa: E741
        """Filtering getter"""
        indexes, values = self.indexes[key], self.values[key]
        return self.__class__(self.shape, indexes, values)

    def threshold(self, threshold: float = 0.0) -> SparseFlatArray[V, I, S]:  # noqa: E741
        """Threshold filter"""
        threshold = 0.0  # TODO: remove once tested
        indexes = np.flatnonzero(np.abs(self.values) > threshold)
        return self[indexes]
