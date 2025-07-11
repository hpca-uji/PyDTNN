"""TenSEAL encryption"""

import sys
import math
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

    def __add__(self, other):
        """Add two ciphertexts"""
        if not isinstance(other, Ciphertext):
            raise NotImplementedError()

        if other._type != self._type:
            raise TypeError(f"Different underlying types ({other._type} != {self._type})")

        if other._shape != self._shape:
            raise TypeError(f"Different underlying shapes ({other._shape} != {self._shape})")

        chunks = tuple(itertools.starmap(operator.add, zip(self._chunks, other._chunks)))

        return Ciphertext(
            _type=self._type,
            _shape=self._shape,
            _chunks=chunks
        )


class Context:
    """TenSEAL context"""

    def __init__(self):
        """Inizialize context"""
        # Context
        self._slots = 4096
        self._private_context = tenseal.context(
            scheme=tenseal.SCHEME_TYPE.CKKS,
            poly_modulus_degree=self._slots * 2,
            coeff_mod_bit_sizes=[40, 40, 40, 40, 40]
        )

        # Keys
        self._private_context.global_scale = 2 ** 40
        self._private_context.generate_galois_keys()
        self._private_context.generate_relin_keys()

        # Public
        self._public_context = self._private_context.copy()
        self._public_context.make_context_public()

    def _pack(self, obj: np.ndarray) -> abc.Generator[list]:
        """Transform numpy array into batched lists"""
        for part in np.array_split(obj.reshape(-1), range(self._slots, obj.size, self._slots)):
            yield part.tolist()

    def _encrypt(self, plain: list) -> CKKSVector:
        """Encode list to ciphertext"""
        return tenseal.ckks_vector(self._public_context, plain)

    def _decrypt(self, cipher: CKKSVector) -> list:
        """Decode cypertext to list"""
        return cipher.decrypt(secret_key=self._private_context.secret_key())

    def encrypt(self, obj: np.ndarray) -> Ciphertext:
        """Encode numpy array to ciphertext"""
        data = tuple(map(self._encrypt, self._pack(obj)))

        return Ciphertext(
            _type=obj.dtype.type,
            _shape=obj.shape,
            _chunks=data
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
    cls = CKKSVector.load
    args = (vector.context(), vector.serialize(),)
    return (cls, args)


copyreg.pickle(SealContext, context_reducer)
copyreg.pickle(CKKSVector, ckks_vector_reducer)
