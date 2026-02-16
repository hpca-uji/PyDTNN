"""uArchFHE encryption"""

# FIXME: Serialization performance

import sys
import copyreg
import dataclasses
from dataclasses import dataclass
from functools import cached_property

import numpy as np

from pydtnn.libs.uhe import core

# Make sure global package is not confused with current package
_pkg = sys.path.pop(0)
try:
    import fhe_py_binding as uarchfhe
finally:
    sys.path.insert(0, _pkg)


__all__ = (
    "Context",
)


@dataclass(eq=False, order=False, slots=True, frozen=True)
class Ciphertext[P: np.number](core.Ciphertext[uarchfhe.PyCiphertext, P]):
    """uArchFHE ciphertext"""
    _context: uarchfhe.PyContext = dataclasses.field(repr=False)

    def _new(self, /, *args, **kwds):
        """Create new operable ciphertext"""
        return super(Ciphertext, self)._new(_context=self._context, *args, **kwds)

    def _operable(self, other) -> None:
        """Ensure ciphertext is operable"""
        super(Ciphertext, self)._operable(other)

        # Synchronize contexts
        self._link_context(self._context)
        other._link_context(self._context)

    def _link_context(self, context: uarchfhe.PyContext) -> None:
        """Link all chunks to context"""
        for chunk in self._chunks:
            chunk.attach_context(context)

    def _add_chunk(self, a: uarchfhe.PyCiphertext, b: uarchfhe.PyCiphertext) -> uarchfhe.PyCiphertext:
        """Add two ciphertexts"""
        return uarchfhe.PyCiphertext.add(a, b)

    def _mul_chunk(self, a: uarchfhe.PyCiphertext, b: uarchfhe.PyCiphertext) -> uarchfhe.PyCiphertext:
        """Multiply two ciphertexts"""
        return uarchfhe.PyCiphertext.mul(a, b)


class Context(core.Context[uarchfhe.PyCiphertext]):
    """uArchFHE context"""
    _cls = Ciphertext

    def __init__(self, options: core.Options = core.Options()) -> None:
        """Initialize context"""
        super().__init__(options)

        # Context
        h = 3  # Secret key Hamming weight (security parameter)
        sigma = 3  # Standard deviation for error distribution (security parameter)
        self._context = uarchfhe.PyContext(self._poly_degree, max(self._coeff_modulus), self._scale, sigma, h)
        self._workspace = [0] * (2 ** self._slots)

        # Keys
        keygen = uarchfhe.PyKeyGen(self._context)
        self._keys = keygen.gen_keys()

    @cached_property
    def _ckks(self) -> uarchfhe.PyCKKS:
        """uArchFHE CKKS context"""
        return uarchfhe.PyCKKS(self._context, self._keys)

    def __getstate__(self) -> object:
        """Get serializable state"""
        state = super().__getstate__()
        state.pop("_ckks", None)
        return state

    def _new(self, /, *args, **kwds) -> core.Ciphertext:
        """Create new operable ciphertext"""
        return super()._new(_context=self._context, *args, **kwds)

    def _encrypt_chunk(self, chunk: list) -> uarchfhe.PyCiphertext:
        """Encode list to ciphertext"""
        if len(chunk) < len(self._workspace):
            self._workspace[:len(chunk)] = chunk
            chunk = self._workspace
        return self._ckks.encrypt(chunk, len(chunk), self._scale, max(self._coeff_modulus))

    def _decrypt_chunk(self, chunk: uarchfhe.PyCiphertext) -> list:
        """Decode cypertext to list"""
        return self._ckks.decrypt(chunk)


# Pickle support
def context_reducer(context: uarchfhe.PyContext) -> tuple:
    """uArchFHE context pickle reducer"""
    cls = context.load_from_memory
    args = (context.save_to_memory(),)
    return (cls, args)


def keychain_reducer(keychain: uarchfhe.PyKeychain) -> tuple:
    """uArchFHE key chain pickle reducer"""
    cls = keychain.load_full_from_memory
    args = (keychain.save_public_keys_to_memory(), keychain.save_secret_key_to_memory())
    return (cls, args)


def ciphertext_reducer(ciphertext: uarchfhe.PyCiphertext) -> tuple:
    """uArchFHE ciphertext reducer"""
    cls = ciphertext.load_from_memory
    args = (ciphertext.save_to_memory(),)
    return (cls, args)


copyreg.pickle(uarchfhe.PyContext, context_reducer)
copyreg.pickle(uarchfhe.PyKeychain, keychain_reducer)
copyreg.pickle(uarchfhe.PyCiphertext, ciphertext_reducer)
