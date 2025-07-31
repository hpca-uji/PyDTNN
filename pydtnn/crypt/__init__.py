"""Abstract encryption"""

import math
import operator
import itertools
import dataclasses
from collections import abc
from dataclasses import dataclass

import numpy as np


__all__ = (
    "Context",
)


@dataclass(eq=False, order=False, slots=True, frozen=True)
class Ciphertext[C, P: np.number]:
    """Abstract ciphertext"""
    dtype: np.dtype[P]
    shape: tuple[int, ...]
    _chunks: tuple[C, ...] = dataclasses.field(repr=False)

    def _new(self, /, *args, **kwds):
        """Create new operable ciphertext"""
        return self.__class__(
            dtype=self.dtype,
            *args, **kwds
        )

    def _operable(self, other) -> None:
        """Ensure ciphertext is operable"""
        if not isinstance(other, Ciphertext):
            raise NotImplementedError()

        if other.dtype != self.dtype:
            raise TypeError(f"Different underlying types ({other.dtype} != {self.dtype})")

    def __add__(self, other):
        """Add two ciphertexts"""
        self._operable(other)

        if other.shape != self.shape:
            raise TypeError(f"Different underlying shapes ({other.shape} != {self.shape})")

        chunks = tuple(itertools.starmap(operator.add, zip(self._chunks, other._chunks)))

        return self._new(
            shape=self.shape,
            _chunks=chunks
        )


class Context[C]:
    """Abstract context"""
    _size = 4096
    _cls: type[Ciphertext]

    def _new[P: np.number](self, /, dtype: np.dtype[P], *args, **kwds) -> Ciphertext[C, P]:
        """Create new operable ciphertext"""
        return self._cls(
            dtype=dtype,
            *args, **kwds
        )

    def _partition(self, obj: np.ndarray) -> abc.Generator[list]:
        """Transform numpy array into batched lists"""
        if obj.size == 0:
            return
        for part in np.array_split(obj.reshape(-1), range(self._size, obj.size, self._size)):
            yield part.tolist()

    def _encrypt_chunk(self, chunk: list) -> C:
        """Encode list to ciphertext"""
        raise NotImplementedError()

    def _decrypt_chunk(self, chunk: C) -> list:
        """Decode cypertext to list"""
        raise NotImplementedError()

    def _encrypt[P: np.number](self, obj: np.ndarray[tuple, np.dtype[P]], /, *args, **kwds) -> Ciphertext[C, P]:
        """Encode numpy array to ciphertext"""
        data = tuple(map(self._encrypt_chunk, self._partition(obj)))

        return self._new(
            dtype=obj.dtype,
            shape=obj.shape,
            _chunks=data,
            *args, **kwds
        )

    def _decrypt[P: np.number](self, obj: Ciphertext[C, P]) -> np.ndarray[tuple, np.dtype[P]]:
        """Decode cypertext to numpy array"""
        data = itertools.chain.from_iterable(map(self._decrypt_chunk, obj._chunks))

        return np.fromiter(
            iter=data,
            dtype=obj.dtype,
            count=math.prod(obj.shape)
        ).reshape(obj.shape)

    def encrypt[P: np.number](self, obj: np.ndarray[tuple, np.dtype[P]]) -> Ciphertext[C, P]:
        """Encode numpy array to ciphertext"""
        return self._encrypt(obj)

    def decrypt[P: np.number](self, obj: Ciphertext[C, P]) -> np.ndarray[tuple, np.dtype[P]]:
        """Decode cypertext to numpy array"""
        return self._decrypt(obj)
