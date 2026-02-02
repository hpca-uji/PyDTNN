"""uArchFHE encryption"""

# FIXME: Serialization performance

from functools import cached_property
import sys
import copyreg
import dataclasses
from dataclasses import dataclass

import numpy as np

from pydtnn import crypto

# Make sure global package is not confused with current package
_pkg = sys.path.pop(0)
try:
    import fhe_py_binding as uarchfhe
finally:
    sys.path.insert(0, _pkg)


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
class Ciphertext[P: np.number](crypto.Ciphertext[uarchfhe.PyCiphertext, P]):
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


class Context(crypto.Context[uarchfhe.PyCiphertext]):
    """uArchFHE context"""
    _cls = Ciphertext

    def __init__(self, poly_degree: int = 13, global_scale: int = 40, security_level: int = 128) -> None:
        """Initialize context"""
        super().__init__(poly_degree, global_scale, security_level)

        # Context
        h = 3  # Secret key Hamming weight (security parameter)
        sigma = 3  # Standard deviation for error distribution (security parameter)
        self._modulus = COEFF_MODULUS[self._security_level][self._poly_degree]
        self._context = uarchfhe.PyContext(self._poly_degree, self._modulus[0], self._global_scale, sigma, h)
        self._workspace = [0] * self._slots

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

    def _new(self, /, *args, **kwds) -> crypto.Ciphertext:
        """Create new operable ciphertext"""
        return super()._new(_context=self._context, *args, **kwds)

    def _encrypt_chunk(self, chunk: list) -> uarchfhe.PyCiphertext:
        """Encode list to ciphertext"""
        if len(chunk) < self._slots:
            self._workspace[:len(chunk)] = chunk
            chunk = self._workspace
        return self._ckks.encrypt(chunk, len(chunk), self._global_scale, self._modulus[0])

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
