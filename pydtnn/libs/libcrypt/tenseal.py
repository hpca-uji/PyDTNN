"""TenSEAL encryption"""

# NOTE: Dataclasses with slots can not use zero-arg super() (gh-90562)
# FIXME: Serialization preformance

import sys
import pickle
import copyreg
import dataclasses
from dataclasses import dataclass

import numpy as np

from pydtnn.libs import libcrypt

# Make sure global package is not confused with current package
_pkg = sys.path.pop(0)
try:
    import tenseal
    from tenseal import sealapi
    from tenseal.tensors import CKKSVector
    from tenseal.enc_context import Context as SealContext
finally:
    sys.path.insert(0, _pkg)


__all__ = (
    "Context",
)


SECURITY_LEVEL = {
    0: sealapi.SEC_LEVEL_TYPE.NONE,
    128: sealapi.SEC_LEVEL_TYPE.TC128,
    192: sealapi.SEC_LEVEL_TYPE.TC192,
    256: sealapi.SEC_LEVEL_TYPE.TC256
}


@dataclass(eq=False, order=False, slots=True, frozen=True)
class Ciphertext[P: np.number](libcrypt.Ciphertext[CKKSVector, P]):
    """TenSEAL ciphertext"""
    _context: bytes = dataclasses.field(repr=False)

    def _new(self, /, *args, **kwds):
        """Create new operable ciphertext"""
        return super(Ciphertext, self)._new(_context=self._context, *args, **kwds)

    def _operable(self, other) -> None:
        """Ensure ciphertext is operable"""
        super(Ciphertext, self)._operable(other)

        if other._context != self._context:
            raise TypeError(f"Different underlying context ({hash(other._context)} != {hash(self._context)})")

        # Synchronize contexts
        context = self._get_context() or other._get_context() or self._load_context()
        self._link_context(context)
        other._link_context(context)

    def _link_context(self, context: SealContext):
        """Link all chunks to context"""
        for chunk in self._chunks:
            chunk.link_context(context)

    def _load_context(self) -> SealContext:
        """Load stored context"""
        return pickle.loads(self._context)

    def _get_context(self) -> SealContext | None:
        """Get loaded context"""
        if not self._chunks:
            return None
        chunk = self._chunks[0]
        try:
            return chunk.context()
        except ValueError:
            return None


class Context(libcrypt.Context[CKKSVector]):
    """TenSEAL context"""
    _cls = Ciphertext

    def __init__(self, poly_degree: int = 13, global_scale: int = 40, security_level: int = 128) -> None:
        """Inizialize context"""
        super().__init__(poly_degree, global_scale, security_level)

        poly_degree = 2 ** self._poly_degree
        level = SECURITY_LEVEL[self._security_level]

        # Context
        modulus = [
            m.bit_count()
            for m in sealapi.CoeffModulus.BFVDefault(poly_degree, level)
        ]
        self._private_context = tenseal.context(
            scheme=tenseal.SCHEME_TYPE.CKKS,
            poly_modulus_degree=poly_degree,
            coeff_mod_bit_sizes=modulus
        )

        # Keys
        self._private_context.global_scale = 2 ** self._global_scale
        self._private_context.generate_galois_keys()
        self._private_context.generate_relin_keys()

        # Public
        self._public_context = self._private_context.copy()
        self._public_context.make_context_public()

        self._context = pickle.dumps(self._public_context)

    def _new(self, /, *args, **kwds) -> libcrypt.Ciphertext:
        """Create new operable ciphertext"""
        return super()._new(_context=self._context, *args, **kwds)

    def _encrypt_chunk(self, chunk: list) -> CKKSVector:
        """Encode list to ciphertext"""
        return tenseal.ckks_vector(self._public_context, chunk)

    def _decrypt_chunk(self, chunk: CKKSVector) -> list:
        """Decode cypertext to list"""
        chunk.link_context(self._private_context)
        return chunk.decrypt()


# Pickle support
def context_reducer(context: SealContext):
    """TenSEAL context pickle reducer"""
    cls = context.load
    args = (context.serialize(save_secret_key=True),)
    return (cls, args)


def ckks_vector_reducer(vector: CKKSVector):
    """TenSEAL CKKS vector pickle reducer"""
    cls = vector.lazy_load
    args = (vector.serialize(),)
    return (cls, args)


copyreg.pickle(SealContext, context_reducer)
copyreg.pickle(CKKSVector, ckks_vector_reducer)  # type: ignore (wrong inferred typing)
