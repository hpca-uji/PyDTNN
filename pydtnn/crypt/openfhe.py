"""OpenFHE encryption"""

import sys
import copyreg
from dataclasses import dataclass

# Make sure global package is not confused with current package
_pkg = sys.path.pop(0)
try:
    import openfhe
finally:
    sys.path.insert(0, _pkg)

import numpy as np

from pydtnn import crypt


__all__ = (
    "Context",
)


@dataclass(repr=False, eq=False, order=False, slots=True, frozen=True)
class Ciphertext[P: np.number](crypt.Ciphertext[openfhe.Ciphertext, P]):
    """OpenFHE ciphertext"""


class Context(crypt.Context[openfhe.Ciphertext]):
    """OpenFHE context"""
    _cls = Ciphertext

    def __init__(self):
        """Inizialize context"""
        ring_dim = self._size * 2
        level = openfhe.SecurityLevel.HEStd_128_classic

        # Context
        parameters = openfhe.CCParamsCKKSRNS()
        parameters.SetSecurityLevel(level)
        parameters.SetRingDim(ring_dim)
        parameters.SetScalingModSize(40)
        parameters.SetMultiplicativeDepth(0)
        self._context = openfhe.GenCryptoContext(parameters)
        self._context.Enable(openfhe.PKESchemeFeature.PKE)
        self._context.Enable(openfhe.PKESchemeFeature.KEYSWITCH)
        self._context.Enable(openfhe.PKESchemeFeature.LEVELEDSHE)

        # Keys
        keys = self._context.KeyGen()
        self._public_key = keys.publicKey
        self._private_key = keys.secretKey

    def _encrypt_chunk(self, chunk: list) -> openfhe.Ciphertext:
        """Encode list to ciphertext"""
        pack = self._context.MakeCKKSPackedPlaintext(chunk)
        cipher = self._context.Encrypt(self._public_key, pack)
        return cipher

    def _decrypt_chunk(self, chunk: openfhe.Ciphertext) -> list:
        """Decode cypertext to list"""
        pack = self._context.Decrypt(chunk, self._private_key)
        plain = pack.GetRealPackedValue()
        return plain


# Serialization
def DeserializeCryptoContext(str: bytes) -> openfhe.CryptoContext:
    """OpenFHE context deserializer"""
    return openfhe.DeserializeCryptoContextString(str, openfhe.BINARY)


def DeserializeCiphertext(str: bytes) -> openfhe.Ciphertext:
    """OpenFHE cipher text deserializer"""
    return openfhe.DeserializeCiphertextString(str, openfhe.BINARY)


def DeserializePublicKey(str: bytes) -> openfhe.PublicKey:
    """OpenFHE public key deserializer"""
    return openfhe.DeserializePublicKeyString(str, openfhe.BINARY)


def DeserializePrivateKey(str: bytes) -> openfhe.PrivateKey:
    """OpenFHE private key deserializer"""
    return openfhe.DeserializePrivateKeyString(str, openfhe.BINARY)


# Pickle support
def context_reducer(context: openfhe.CryptoContext):
    """OpenFHE context pickle reducer"""
    cls = DeserializeCryptoContext
    args = (openfhe.Serialize(context, openfhe.BINARY),)
    return (cls, args)


def ciphertext_reducer(ciphertext: openfhe.Ciphertext):
    """OpenFHE cipher text pickle reducer"""
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
