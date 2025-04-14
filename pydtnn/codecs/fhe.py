"""Fully Homomorphic Encryption codec"""

# NOTE: Module does not provide any encryption, it is temporary placeholder

# TODO: Move to libs and remove codecs

import numpy
from dataclasses import dataclass


__all__ = (
    "Codec",
)


@dataclass(repr=False, eq=False, order=False, slots=True, frozen=True)
class Ciphertext:
    """Pseudo-ciphertext"""
    _plain: numpy.ndarray

    def __add__(self, other):
        """Add two ciphertexts"""
        if not isinstance(other, Ciphertext):
            raise NotImplementedError()
        return Ciphertext(self._plain + other._plain)

    def __mul__(self, other):
        """Multiply two ciphertexts"""
        if not isinstance(other, Ciphertext):
            raise NotImplementedError()
        return Ciphertext(self._plain * other._plain)


class Codec:
    """Fully Homomorphic Encryption codec"""

    def encode(self, obj: numpy.ndarray) -> Ciphertext:
        """Encode object to ciphertext"""
        return Ciphertext(obj)

    def decode(self, obj: Ciphertext) -> numpy.ndarray:
        """Decode cypertext to object"""
        return obj._plain
