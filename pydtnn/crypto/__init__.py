"""Abstract encryption"""

# NOTE: HELib: ploy = slots + 2

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


# COEFF_MODULUS[security_level][poly_degree]
# source: sealapi.CoeffModulus.BFVDefault
COEFF_MODULUS = {
    128: {
        10: [27],
        11: [54],
        12: [36, 36, 37],
        13: [43, 43, 44, 44, 44],
        14: [48, 48, 48, 49, 49, 49, 49, 49, 49],
        15: [55, 55, 55, 55, 55, 55, 55, 55, 55, 55, 55, 55, 55, 55, 55, 56]
    },
    192: {
        10: [19],
        11: [37],
        12: [25, 25, 25],
        13: [38, 38, 38, 38],
        14: [50, 50, 50, 50, 50, 50],
        15: [54, 54, 54, 54, 54, 55, 55, 55, 55, 55, 55]
    },
    256: {
        10: [14],
        11: [29],
        12: [58],
        13: [39, 39, 40],
        14: [47, 47, 47, 48, 48],
        15: [52, 53, 53, 53, 53, 53, 53, 53, 53]
    }
}


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

    def _add_chunk(self, a: C, b: C) -> C:
        """Add two ciphertexts"""
        return operator.add(a, b)

    def _mul_chunk(self, a: C, b: C) -> C:
        """Multiply two ciphertexts"""
        return operator.mul(a, b)

    def _op_scalar(self, op, other):
        """Execute a element by element operation"""
        self._operable(other)

        if other.shape != self.shape:
            raise TypeError(f"Different underlying shapes ({other.shape} != {self.shape})")

        chunks = tuple(itertools.starmap(op, zip(self._chunks, other._chunks)))

        return self._new(
            shape=self.shape,
            _chunks=chunks
        )

    def __add__(self, other):
        """Add two ciphertexts"""
        return self._op_scalar(self._add_chunk, other)

    def __mul__(self, other):
        """Multiply two ciphertexts"""
        return self._op_scalar(self._mul_chunk, other)


class Context[C]:
    """Abstract context"""
    _cls: type[Ciphertext]

    def __init__(self, slots: int = 12, scale: int = 40, security: int = 128) -> None:
        """Initialize context"""
        self._slots = slots
        self._scale = scale
        self._security = security

    @property
    def _poly_degree(self) -> int:
        """Polynomial degree"""
        return self._slots + 1

    @property
    def _coeff_modulus(self) -> list[int]:
        """Coefficient modulus"""
        return COEFF_MODULUS[self._security][self._poly_degree]

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

        chunk_size = 2 ** self._slots

        for part in np.array_split(obj.reshape(-1), range(chunk_size, obj.size, chunk_size)):
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
