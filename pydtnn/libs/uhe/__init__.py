"""uHE package"""

import enum
import importlib

from pydtnn.libs.uhe.core import Options, Context, Ciphertext


__all__ = (
    "new",
    "Backend",
    "Options",
    "Context",
    "Ciphertext"
)


class Backend(enum.StrEnum):
    """Encryption backend"""
    OPENFHE = enum.auto()
    TENSEAL = enum.auto()
    UARCHFHE = enum.auto()


def new(backend: Backend = Backend.OPENFHE, options: Options = Options()) -> Context:
    """Generate encryption context"""
    module = importlib.import_module(f"{__name__}.{backend}")
    cls: type[Context] = getattr(module, "Context")
    return cls(options)
