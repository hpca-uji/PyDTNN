"""uArchFHE encryption"""

# FIXME: Serialization performance

import sys
import copyreg
import tempfile
import dataclasses
from pathlib import Path
from dataclasses import dataclass

import numpy as np

from pydtnn.libs import libcrypt

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
class Ciphertext[P: np.number](libcrypt.Ciphertext[uarchfhe.PyCiphertext, P]):
    """uArchFHE ciphertext"""
    _context: uarchfhe.PyContext = dataclasses.field(repr=False)

    def _new(self, /, *args, **kwds):
        """Create new operable ciphertext"""
        return super(Ciphertext, self)._new(_context=self._context, *args, **kwds)

    def _add_chunk(self, a: uarchfhe.PyCiphertext, b: uarchfhe.PyCiphertext) -> uarchfhe.PyCiphertext:
        """Add two ciphertexts"""
        return uarchfhe.PyCiphertext.add(a, b)

    @classmethod
    def _expand(cls, dtype, shape, chunks, context):
        """Deserialize ciphertext"""
        chunks = tuple(pyciphertext_load_bytes(chunk, context) for chunk in chunks)
        return cls(dtype, shape, chunks, context)

    def __reduce__(self) -> tuple:
        """Serialize ciphertext"""
        cls = self._expand
        args = (self.dtype, self.shape, tuple(pyciphertext_save_bytes(chunk) for chunk in self._chunks), self._context)
        return (cls, args)


class Context(libcrypt.Context[uarchfhe.PyCiphertext]):
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

        # Keys
        keygen = uarchfhe.PyKeyGen(self._context)
        self._keys = keygen.gen_keys()

    def _new(self, /, *args, **kwds) -> libcrypt.Ciphertext:
        """Create new operable ciphertext"""
        return super()._new(_context=self._context, *args, **kwds)

    def _encrypt_chunk(self, chunk: list) -> uarchfhe.PyCiphertext:
        """Encode list to ciphertext"""
        ckks = uarchfhe.PyCKKS(self._context, self._keys)
        chunk = chunk + [chunk[0]] * (self._slots - len(chunk))
        return ckks.encrypt(chunk, len(chunk), self._global_scale, self._modulus[0])

    def _decrypt_chunk(self, chunk: uarchfhe.PyCiphertext) -> list:
        """Decode cypertext to list"""
        ckks = uarchfhe.PyCKKS(self._context, self._keys)
        return ckks.decrypt(chunk)


# Serialization
def pycontext_load_bytes(data: bytes) -> uarchfhe.PyContext:
    """uArchFHE context deserializer"""
    with tempfile.TemporaryDirectory() as dir:
        path = Path(dir, __name__)
        path.write_bytes(data)
        return uarchfhe.PyContext.load(str(path))


def pycontext_save_bytes(context: uarchfhe.PyContext) -> bytes:
    """uArchFHE context serializer"""
    with tempfile.TemporaryDirectory() as dir:
        path = Path(dir, __name__)
        context.save(str(path))
        return path.read_bytes()


def pykeychain_load_full_bytes(data: bytes, private_key: bytes | None = None) -> uarchfhe.PyKeychain:
    """uArchFHE key chain deserializer"""
    with tempfile.TemporaryDirectory() as dir:
        path = Path(dir, __name__)
        path.write_bytes(data)
        if private_key is None:
            secret_path = None
        else:
            secret_path = Path(dir, f"{__name__}.private")
            secret_path.write_bytes(private_key)
        return uarchfhe.PyKeychain.load_full(str(path), None if private_key is None else str(secret_path), use_json=False)


def pykeychain_save_public_bytes(keychain: uarchfhe.PyKeychain) -> bytes:
    """uArchFHE key chain serializer"""
    with tempfile.TemporaryDirectory() as dir:
        path = Path(dir, __name__)
        keychain.save_public(str(path), use_json=False)
        return path.read_bytes()


def pysecretkey_load_bytes(data: bytes) -> uarchfhe.PySecretKey:
    """uArchFHE private key deserializer"""
    with tempfile.TemporaryDirectory() as dir:
        path = Path(dir, __name__)
        path.write_bytes(data)
        return uarchfhe.PySecretKey.load(str(path))


def pysecretkey_save_bytes(private_key: uarchfhe.PySecretKey) -> bytes:
    """uArchFHE private key serializer"""
    with tempfile.TemporaryDirectory() as dir:
        path = Path(dir, __name__)
        private_key.save(str(path))
        return path.read_bytes()


def pyciphertext_load_bytes(data: bytes, context: uarchfhe.PyContext) -> uarchfhe.PyCiphertext:
    """uArchFHE cipher text deserializer"""
    with tempfile.TemporaryDirectory() as dir:
        path = Path(dir, __name__)
        path.write_bytes(data)
        return uarchfhe.PyCiphertext.load(str(path), context)


def pyciphertext_save_bytes(ciphertext: uarchfhe.PyCiphertext) -> bytes:
    """uArchFHE cipher text serializer"""
    with tempfile.TemporaryDirectory() as dir:
        path = Path(dir, __name__)
        ciphertext.save(str(path))
        return path.read_bytes()


# Pickle support
def context_reducer(context: uarchfhe.PyContext):
    """uArchFHE context pickle reducer"""
    cls = pycontext_load_bytes
    args = (pycontext_save_bytes(context),)
    return (cls, args)


def keychain_reducer(keychain: uarchfhe.PyKeychain):
    """uArchFHE key chain pickle reducer"""
    cls = pykeychain_load_full_bytes
    args = (pykeychain_save_public_bytes(keychain), pysecretkey_save_bytes(keychain.secret))
    return (cls, args)


copyreg.pickle(uarchfhe.PyContext, context_reducer)
copyreg.pickle(uarchfhe.PyKeychain, keychain_reducer)
