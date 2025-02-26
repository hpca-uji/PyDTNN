"""numpy array codec"""

import io
import numpy


__all__ = (
    "Codec",
)


class Codec:
    """numpy array codec"""

    def encode(self, obj: numpy.ndarray) -> bytes:
        """Encode array to bytes"""
        with io.BytesIO() as buffer:
            numpy.save(buffer, obj, allow_pickle=False)
            return buffer.getvalue()

    def decode(self, obj: bytes) -> numpy.ndarray:
        """Decode bytes to array"""
        with io.BytesIO(obj) as buffer:
            return numpy.load(buffer, allow_pickle=False)
