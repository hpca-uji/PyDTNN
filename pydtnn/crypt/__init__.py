"""Abstract encryption"""

# TODO: Migrate to pyfhel

import math
import operator
import itertools
import dataclasses
from collections import abc
from dataclasses import dataclass

import numpy as np
from pydtnn.utils.constants import ArrayShape


__all__ = (
    "Context",
)


@dataclass(eq=False, order=False, slots=True, frozen=True)
class Ciphertext[C, P: np.number]:
    """Abstract ciphertext"""
    dtype: np.dtype[P]
    shape: ArrayShape
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
    _cls: type[Ciphertext]

    def __init__(self, poly_degree: int = 13, global_scale: int = 40, security_level: int = 128) -> None:
        """Inizialize context"""
        self._poly_degree = poly_degree
        self._global_scale = global_scale
        self._security_level = security_level

    @property
    def _slots(self) -> int:
        return 2 ** (self._poly_degree // 2)

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
        for part in np.array_split(obj.reshape(-1), range(self._slots, obj.size, self._slots)):
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
