"""Python object codec"""

import pickle


__all__ = (
    "Codec",
)


class Codec:
    """Python object codec"""
    _protocol = 5

    def encode(self, obj) -> bytes:
        """Encode object to bytes"""
        return pickle.dumps(obj, protocol=self._protocol)

    def decode(self, obj: bytes):
        """Decode bytes to object"""
        return pickle.loads(obj)
