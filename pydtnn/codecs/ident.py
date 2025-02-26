"""Identity codec"""


__all__ = (
    "Codec",
)


class Codec:
    """Identity codec"""

    def encode[T](self, obj: T) -> T:
        """Encode object as itself"""
        return obj

    def decode[T](self, obj: T) -> T:
        """Decode object as itself"""
        return obj
