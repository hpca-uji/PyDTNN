"""OpenFHE encryption"""

import sys
import math
import copyreg
import operator
import itertools
from collections import abc
from dataclasses import dataclass

# Make sure global package is not confused with current package
_pkg = sys.path.pop(0)
try:
    import openfhe  # noqa: F401
finally:
    sys.path.insert(0, _pkg)

import numpy as np


__all__ = (
    "Context",
)


@dataclass(repr=False, eq=False, order=False, slots=True, frozen=True)
class Ciphertext[C, P: np.number]:
    """Abstract ciphertext"""
    _type: P
    _shape: tuple[int, ...]
    _chunks: tuple[C, ...]

    def __add__(self, other):
        """Add two ciphertexts"""
        if not isinstance(other, Ciphertext):
            raise NotImplementedError()

        if other._type != self._type:
            raise TypeError(f"Different underlying types ({other._type} != {self._type})")

        if other._shape != self._shape:
            raise TypeError(f"Different underlying shapes ({other._shape} != {self._shape})")

    def _add(self, other, /, *args, **kwds):
        """Add two ciphertexts"""
        chunks = tuple(itertools.starmap(operator.add, zip(self._chunks, other._chunks)))

        return Ciphertext(
            _type=self._type,
            _shape=self._shape,
            _chunks=chunks,
            *args, **kwds
        )


class Context[C, P: np.number]:
    """Abstract context"""
    _slots = 4096

    def __init__(self):
        """Inizialize context"""

    def _chunk_array(self, obj: np.ndarray[tuple, np.dtype[P]]) -> abc.Generator[list]:
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

    def _encrypt(self, obj: np.ndarray[tuple, np.dtype[P]], /, *args, **kwds) -> Ciphertext[C, P]:
        """Encode numpy array to ciphertext"""
        data = tuple(map(self._encrypt_chunk, self._chunk_array(obj)))

        return Ciphertext(
            _type=obj.dtype.type,
            _shape=obj.shape,
            _chunks=data,
            *args, **kwds
        )

    def _decrypt(self, obj: Ciphertext[C, P]) -> np.ndarray[tuple, np.dtype[P]]:
        """Decode cypertext to numpy array"""
        data = itertools.chain.from_iterable(map(self._decrypt_chunk, obj._chunks))

        return np.fromiter(
            iter=data,
            dtype=obj._type,
            count=math.prod(obj._shape)
        ).reshape(obj._shape)

    def encrypt(self, obj: np.ndarray[tuple, np.dtype[P]]) -> Ciphertext[C, P]:
        """Encode numpy array to ciphertext"""
        return self._encrypt(obj)

    def decrypt(self, obj: Ciphertext[C, P]) -> np.ndarray[tuple, np.dtype[P]]:
        """Decode cypertext to numpy array"""
        return self._decrypt(obj)
