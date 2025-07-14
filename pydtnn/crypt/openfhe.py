"""OpenFHE encryption"""

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
    import openfhe  # noqa: F401
finally:
    sys.path.insert(0, _pkg)

import numpy as np


__all__ = (
    "Context",
)


@dataclass(repr=False, eq=False, order=False, slots=True, frozen=True)
class Ciphertext:
    """OpenFHE ciphertext"""
    _type: np.number
    _shape: tuple[int, ...]
    _chunks: tuple[openfhe.Ciphertext, ...]

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
    """OpenFHE context"""

    def __init__(self):
        """Inizialize context"""

        # Context
        self._slots = 4096
        parameters = openfhe.CCParamsCKKSRNS()
        parameters.SetScalingModSize(40)
        parameters.SetMultiplicativeDepth(0)
        parameters.SetRingDim(self._slots * 2)
        self._context = openfhe.GenCryptoContext(parameters)
        self._context.Enable(openfhe.PKESchemeFeature.PKE)
        self._context.Enable(openfhe.PKESchemeFeature.KEYSWITCH)
        self._context.Enable(openfhe.PKESchemeFeature.LEVELEDSHE)

        # Keys
        keys = self._context.KeyGen()
        self._public_key = keys.publicKey
        self._private_key = keys.secretKey

    def _pack(self, obj: np.ndarray) -> abc.Generator[list]:
        """Transform numpy array into batched lists"""
        for part in np.array_split(obj.reshape(-1), range(self._slots, obj.size, self._slots)):
            yield part.tolist()

    def _encrypt(self, plain: list) -> openfhe.Ciphertext:
        """Encode list to ciphertext"""
        pack = self._context.MakeCKKSPackedPlaintext(plain)
        cipher = self._context.Encrypt(self._public_key, pack)
        return cipher

    def _decrypt(self, cipher: openfhe.Ciphertext) -> list:
        """Decode cypertext to list"""
        pack = self._context.Decrypt(cipher, self._private_key)
        plain = pack.GetRealPackedValue()
        return plain

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


# Serialization
def DeserializeCryptoContext(str: bytes) -> openfhe.CryptoContext:
    """OpenFHE context deserializer"""
    return openfhe.DeserializeCryptoContextString(str, openfhe.BINARY)


def DeserializeCiphertext(str: bytes) -> openfhe.Ciphertext:
    """OpenFHE context deserializer"""
    return openfhe.DeserializeCiphertextString(str, openfhe.BINARY)


def DeserializePublicKey(str: bytes) -> openfhe.PublicKey:
    """OpenFHE context deserializer"""
    return openfhe.DeserializePublicKeyString(str, openfhe.BINARY)


def DeserializePrivateKey(str: bytes) -> openfhe.PrivateKey:
    """OpenFHE context deserializer"""
    return openfhe.DeserializePrivateKeyString(str, openfhe.BINARY)


# Pickle support
def context_reducer(context: openfhe.CryptoContext):
    """OpenFHE context pickle reducer"""
    cls = DeserializeCryptoContext
    args = (openfhe.Serialize(context, openfhe.BINARY),)
    return (cls, args)


def ciphertext_reducer(ciphertext: openfhe.Ciphertext):
    """OpenFHE ciphertext pickle reducer"""
    cls = DeserializeCiphertext
    args = (openfhe.Serialize(ciphertext, openfhe.BINARY),)
    return (cls, args)


def public_key_reducer(ciphertext: openfhe.PublicKey):
    """OpenFHE public key pickle reducer"""
    cls = DeserializePublicKey
    args = (openfhe.Serialize(ciphertext, openfhe.BINARY),)
    return (cls, args)


def private_key_reducer(ciphertext: openfhe.PrivateKey):
    """OpenFHE private key pickle reducer"""
    cls = DeserializePrivateKey
    args = (openfhe.Serialize(ciphertext, openfhe.BINARY),)
    return (cls, args)


copyreg.pickle(openfhe.CryptoContext, context_reducer)
copyreg.pickle(openfhe.Ciphertext, ciphertext_reducer)
copyreg.pickle(openfhe.PublicKey, public_key_reducer)
copyreg.pickle(openfhe.PrivateKey, private_key_reducer)
