"""Ciphertext codec"""

from dataclasses import dataclass


__all__ = (
    "Codec",
)


@dataclass(eq=False, order=False, slots=True, frozen=True)
class Ciphertext[T]:
    """Pseudo-ciphertext"""
    plain: T

    def __add__(self, other):
        """Add two ciphertexts"""
        if not isinstance(other, Ciphertext):
            raise NotImplementedError()
        return Ciphertext[T](self.plain + other.plain)

    def __mul__(self, other):
        """Multiply two ciphertexts"""
        if not isinstance(other, Ciphertext):
            raise NotImplementedError()
        return Ciphertext[T](self.plain * other.plain)


class Codec:
    """Ciphertext codec"""

    def encode[T](self, obj: T) -> Ciphertext[T]:
        """Encode object to ciphertext"""
        return Ciphertext(obj)

    def decode[T](self, obj: Ciphertext[T]) -> T:
        """Decode cypertext to object"""
        return obj.plain
