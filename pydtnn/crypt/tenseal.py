"""TenSEAL encryption"""

import sys
import math
import pickle
import copyreg
import operator
import itertools
from collections import abc
from dataclasses import dataclass

# Make sure global package is not confused with current package
_pkg = sys.path.pop(0)
try:
    import tenseal  # noqa: F401
finally:
    sys.path.insert(0, _pkg)

import numpy as np
from tenseal import sealapi
from tenseal.tensors import CKKSVector
from tenseal.enc_context import Context as SealContext


__all__ = (
    "Context",
)


@dataclass(repr=False, eq=False, order=False, slots=True, frozen=True)
class Ciphertext:
    """TenSEAL ciphertext"""
    _type: np.number
    _shape: tuple[int, ...]
    _chunks: tuple[CKKSVector, ...]
    _context: bytes

    def __add__(self, other):
        """Add two ciphertexts"""
        if not isinstance(other, Ciphertext):
            raise NotImplementedError()

        if other._type != self._type:
            raise TypeError(f"Different underlying types ({other._type} != {self._type})")

        if other._shape != self._shape:
            raise TypeError(f"Different underlying shapes ({other._shape} != {self._shape})")

        if other._context != self._context:
            raise TypeError(f"Different underlying contexts ({hash(other._context)} != {hash(self._context)})")

        # Relink contexts
        context = self._get_context() or other._get_context() or self._load_context()
        for chunk in itertools.chain(self._chunks, other._chunks):
            chunk.link_context(context)

        chunks = tuple(itertools.starmap(operator.add, zip(self._chunks, other._chunks)))

        return Ciphertext(
            _type=self._type,
            _shape=self._shape,
            _chunks=chunks,
            _context=self._context
        )

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


class Context:
    """TenSEAL context"""

    def __init__(self):
        """Inizialize context"""
        self._slots = 4096
        poly_degree = self._slots * 2
        level = sealapi.SEC_LEVEL_TYPE.TC128

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
        self._private_context.global_scale = 2 ** 40
        self._private_context.generate_galois_keys()
        self._private_context.generate_relin_keys()

        # Public
        self._public_context = self._private_context.copy()
        self._public_context.make_context_public()

        self._context = pickle.dumps(self._public_context)

    def _pack(self, obj: np.ndarray) -> abc.Generator[list]:
        """Transform numpy array into batched lists"""
        if obj.size == 0:
            return
        for part in np.array_split(obj.reshape(-1), range(self._slots, obj.size, self._slots)):
            yield part.tolist()

    def _encrypt(self, plain: list) -> CKKSVector:
        """Encode list to ciphertext"""
        return tenseal.ckks_vector(self._public_context, plain)

    def _decrypt(self, cipher: CKKSVector) -> list:
        """Decode cypertext to list"""
        cipher.link_context(self._private_context)
        return cipher.decrypt()

    def encrypt(self, obj: np.ndarray) -> Ciphertext:
        """Encode numpy array to ciphertext"""
        data = tuple(map(self._encrypt, self._pack(obj)))

        return Ciphertext(
            _type=obj.dtype.type,
            _shape=obj.shape,
            _chunks=data,
            _context=self._context
        )

    def decrypt(self, obj: Ciphertext) -> np.ndarray:
        """Decode cypertext to numpy array"""
        data = itertools.chain.from_iterable(map(self._decrypt, obj._chunks))

        return np.fromiter(
            iter=data,
            dtype=obj._type,
            count=math.prod(obj._shape)
        ).reshape(obj._shape)


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
copyreg.pickle(CKKSVector, ckks_vector_reducer)
