"""OpenFHE encryption"""

import sys
import math
import copyreg
import itertools
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
    _data: list[openfhe.Ciphertext]
    _context: openfhe.CryptoContext

    def __add__(self, other):
        """Add two ciphertexts"""
        if not isinstance(other, Ciphertext):
            raise NotImplementedError()

        if other._type != self._type:
            raise TypeError(f"Different underlying types ({other._type} != {self._type})")

        if other._shape != self._shape:
            raise TypeError(f"Different underlying shapes ({other._shape} != {self._shape})")

        data = list(itertools.starmap(self._context.EvalAdd, zip(self._data, other._data)))

        return Ciphertext(
            _type=self._type,
            _shape=self._shape,
            _data=data,
            _context=self._context
        )


class Context:
    """OpenFHE context"""

    def __init__(self):
        """Inizialize context"""

        # Context
        self._slots = 8192
        config = openfhe.CCParamsCKKSRNS()
        config.SetScalingModSize(40)
        config.SetRingDim(self._slots * 2)
        self._context = openfhe.GenCryptoContext(config)
        self._context.Enable(openfhe.PKESchemeFeature.PKE)
        self._context.Enable(openfhe.PKESchemeFeature.KEYSWITCH)
        self._context.Enable(openfhe.PKESchemeFeature.LEVELEDSHE)

        # Keys
        keys = self._context.KeyGen()
        self._public_key = keys.publicKey
        self._private_key = keys.secretKey

    def encrypt(self, obj: np.ndarray) -> Ciphertext:
        """Encode object to ciphertext"""
        raws = np.split(obj.reshape(-1), range(self._slots, obj.size, self._slots))
        plains = [self._context.MakeCKKSPackedPlaintext(raw.tolist()) for raw in raws]
        ciphers = [self._context.Encrypt(self._public_key, plain) for plain in plains]

        return Ciphertext(
            _type=obj.dtype.type,
            _shape=obj.shape,
            _data=ciphers,
            _context=self._context
        )

    def decrypt(self, obj: Ciphertext) -> np.ndarray:
        """Decode cypertext to object"""
        plains = [self._context.Decrypt(data, self._private_key) for data in obj._data]
        raws = [plain.GetRealPackedValue() for plain in plains]
        data = list(itertools.islice(itertools.chain.from_iterable(raws), math.prod(obj._shape)))

        return np.array(
            object=data,
            dtype=obj._type
        ).reshape(obj._shape)


# Deserialization
def DeserializeCryptoContext(str: bytes):
    """OpenFHE context deserializer"""
    return openfhe.DeserializeCryptoContextString(str, openfhe.BINARY)


def DeserializeCiphertext(str: bytes):
    """OpenFHE context deserializer"""
    return openfhe.DeserializeCiphertextString(str, openfhe.BINARY)


def DeserializePublicKey(str: bytes):
    """OpenFHE context deserializer"""
    return openfhe.DeserializePublicKeyString(str, openfhe.BINARY)


def DeserializePrivateKey(str: bytes):
    """OpenFHE context deserializer"""
    return openfhe.DeserializePrivateKeyString(str, openfhe.BINARY)


# Serialization
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
