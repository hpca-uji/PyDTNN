"""TenSEAL encryption"""

import sys
import copyreg
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
    _data: CKKSVector

    def __add__(self, other):
        """Add two ciphertexts"""
        if not isinstance(other, Ciphertext):
            raise NotImplementedError()

        if other._type != self._type:
            raise TypeError(f"Different underlying types ({other._type} != {self._type})")

        if other._shape != self._shape:
            raise TypeError(f"Different underlying shapes ({other._shape} != {self._shape})")

        data = self._data.add(other._data)

        return Ciphertext(
            _type=self._type,
            _shape=self._shape,
            _data=data
        )


class Context:
    """TenSEAL context"""

    def __init__(self):
        """Inizialize context"""
        # Context
        self._slots = 8192
        self._private_context = tenseal.context(
            scheme=tenseal.SCHEME_TYPE.CKKS,
            poly_modulus_degree=self._slots,
            coeff_mod_bit_sizes=[40, 20, 20, 20, 40]
        )

        # Keys
        self._private_context.global_scale = 2 ** 20
        self._private_context.generate_galois_keys()
        self._private_context.generate_relin_keys()

        # Public
        self._public_context = self._private_context.copy()
        self._public_context.make_context_public()

    def encrypt(self, obj: np.ndarray) -> Ciphertext:
        """Encode object to ciphertext"""
        data = tenseal.ckks_vector(self._private_context, obj.flat)
        data.link_context(self._public_context)

        return Ciphertext(
            _type=obj.dtype.type,
            _shape=obj.shape,
            _data=data
        )

    def decrypt(self, obj: Ciphertext) -> np.ndarray:
        """Decode cypertext to object"""
        data = obj._data.decrypt(secret_key=self._private_context.secret_key())

        return np.array(
            object=data,
            dtype=obj._type
        ).reshape(obj._shape)


# Serialization
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
