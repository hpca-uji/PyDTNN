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
from tenseal.enc_context import Context as SealContext
from tenseal.tensors.abstract_tensor import AbstractTensor
from tenseal.tensors import CKKSVector, BFVVector, CKKSTensor, BFVTensor


__all__ = (
    "Context",
)


@dataclass(repr=False, eq=False, order=False, slots=True, frozen=True)
class Ciphertext:
    """TenSEAL ciphertext"""
    _type: np.number
    _shape: tuple[int, ...]
    _data: AbstractTensor

    def __add__(self, other):
        """Add two ciphertexts"""
        if not isinstance(other, Ciphertext):
            raise NotImplementedError()

        if other._type != self._type:
            raise TypeError(f"Different underlying types ({other._type} != {self._type})")

        if other._shape != self._shape:
            raise TypeError(f"Different underlying shapes ({other._shape} != {self._shape})")

        return Ciphertext(
            _type=self._type,
            _shape=self._shape,
            _data=self._data + other._data
        )


class Context:
    """TenSEAL context"""

    def __init__(self):
        """Inizialize context"""
        self._scheme = tenseal.SCHEME_TYPE.CKKS

        # Context
        self._context = tenseal.context(
            scheme=self._scheme,
            poly_modulus_degree=8192,
            coeff_mod_bit_sizes=[40, 20, 20, 20, 40]
        )

        # Keys
        self._context.global_scale = 2**20
        self._context.generate_galois_keys()
        self._context.generate_relin_keys()

        # Public
        self._public_context = self._context.copy()
        self._public_context.make_context_public()

    def encrypt(self, obj: np.ndarray) -> Ciphertext:
        """Encode object to ciphertext"""

        match self._scheme:
            case tenseal.SCHEME_TYPE.CKKS:
                data = tenseal.ckks_vector(self._context, obj.flat)
            case tenseal.SCHEME_TYPE.BFV:
                data = tenseal.bfv_vector(self._context, obj.flat)
            case _:
                raise TypeError(f"Unsupported scheme {self._scheme}")

        data.link_context(self._public_context)

        return Ciphertext(
            _type=obj.dtype.type,
            _shape=obj.shape,
            _data=data
        )

    def decrypt(self, obj: Ciphertext) -> np.ndarray:
        """Decode cypertext to object"""
        return np.array(
            object=obj._data._decrypt(secret_key=self._context.secret_key()),
            dtype=obj._type
        ).reshape(obj._shape)


# Serialization
def context_reducer(context: SealContext):
    """TenSEAL context pickle reducer"""
    cls = context.load
    args = (context.serialize(save_secret_key=True),)
    return (cls, args)


def tensor_reducer(tensor: AbstractTensor):
    """TenSEAL tensor pickle reducer"""
    cls = tensor.load
    args = (tensor.context(), tensor.serialize(),)
    return (cls, args)


copyreg.pickle(SealContext, context_reducer)
copyreg.pickle(BFVVector, tensor_reducer)
copyreg.pickle(CKKSVector, tensor_reducer)
copyreg.pickle(BFVTensor, tensor_reducer)
copyreg.pickle(CKKSTensor, tensor_reducer)
