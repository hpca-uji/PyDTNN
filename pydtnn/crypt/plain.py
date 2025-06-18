"""Pseudo encryption"""

import numpy
from dataclasses import dataclass


__all__ = (
    "Context",
)


@dataclass(repr=False, eq=False, order=False, slots=True, frozen=True)
class Ciphertext:
    """Pseudo-ciphertext"""
    _data: numpy.ndarray

    def __add__(self, other):
        """Add two ciphertexts"""
        if not isinstance(other, Ciphertext):
            raise NotImplementedError()

        return Ciphertext(self._data + other._data)


class Context:
    """Pseudo-context"""

    def encrypt(self, obj: numpy.ndarray) -> Ciphertext:
        """Encode object to ciphertext"""
        return Ciphertext(obj)

    def decrypt(self, obj: Ciphertext) -> numpy.ndarray:
        """Decode cypertext to object"""
        return obj._data
