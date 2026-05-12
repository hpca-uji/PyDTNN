"""Matrix multiplication utilities for PyDTNN using BLIS and MKL backends."""
import ctypes
import logging

import numpy as np

from pydtnn.utils import load_library

__all__ = (
    "blis",
    "matmul",
    "matmul_blis",
    "matmul_mkl",
    "mkl",
)

logger = logging.getLogger(__name__)


def blis():
    """Lazy load and return the BLIS library instance."""
    if not hasattr(blis, "lib"):
        blis.lib = load_library("blis")
    return blis.lib


def mkl():
    """Lazy load and return the MKL library instance."""
    if not hasattr(mkl, "lib"):
        mkl.lib = load_library("mkl_rt")
    return mkl.lib


def _matmul_xgemm(called_from, lib, a, b, c=None):
    """Internal helper to perform matrix multiplication using BLAS xGEMM interface."""
    order = 101  # 101 for row-major, 102 for column major data structures
    m = a.shape[0]
    n = b.shape[1]
    k = a.shape[1]
    if c is None:
        c = np.ones((m, n), a.dtype)
    # trans_{a,b} = 111 for no transpose, 112 for transpose, and 113 for conjugate transpose
    if a.flags["C_CONTIGUOUS"]:
        trans_a = 111
        lda = k
    elif a.flags["F_CONTIGUOUS"]:
        trans_a = 112
        lda = m
    else:
        raise ValueError(f"Matrix a data layout not supported by {called_from}().")
    if b.flags["C_CONTIGUOUS"]:
        trans_b = 111
        ldb = n
    elif b.flags["F_CONTIGUOUS"]:
        trans_b = 112
        ldb = k
    else:
        raise ValueError(f"Matrix b data layout not supported by {called_from}().")
    ldc = n
    alpha = 1.0
    beta = 0.0
    if a.dtype == np.float32:
        lib.cblas_sgemm(
            ctypes.c_int(order),
            ctypes.c_int(trans_a),
            ctypes.c_int(trans_b),
            ctypes.c_int(m),
            ctypes.c_int(n),
            ctypes.c_int(k),
            ctypes.c_float(alpha),
            ctypes.c_void_p(a.ctypes.data),
            ctypes.c_int(lda),
            ctypes.c_void_p(b.ctypes.data),
            ctypes.c_int(ldb),
            ctypes.c_float(beta),
            ctypes.c_void_p(c.ctypes.data),
            ctypes.c_int(ldc),
        )
    elif a.dtype == np.float64:
        lib.cblas_dgemm(
            ctypes.c_int(order),
            ctypes.c_int(trans_a),
            ctypes.c_int(trans_b),
            ctypes.c_int(m),
            ctypes.c_int(n),
            ctypes.c_int(k),
            ctypes.c_double(alpha),
            ctypes.c_void_p(a.ctypes.data),
            ctypes.c_int(lda),
            ctypes.c_void_p(b.ctypes.data),
            ctypes.c_int(ldb),
            ctypes.c_double(beta),
            ctypes.c_void_p(c.ctypes.data),
            ctypes.c_int(ldc),
        )
    else:
        raise TypeError(f"Type '{a.dtype}' not supported by {called_from}().")
    return c


def matmul_mkl(a, b, c=None):
    """Perform matrix multiplication using the MKL backend."""
    # os.environ['GOMP_CPU_AFFINITY'] = ""
    # os.environ['OMP_PLACES'] = ""
    return _matmul_xgemm("matmul_mkl", mkl(), a, b, c)


def matmul_blis(a, b, c=None):
    """Perform matrix multiplication using the BLIS backend."""
    return _matmul_xgemm("matmul_blis", blis(), a, b, c)


# Matmul operation
# Warning: the output matrix can not be cached, as it will persist outside this method
def matmul(a: np.ndarray, b: np.ndarray, c: np.ndarray | None = None) -> np.ndarray:
    """Perform matrix multiplication using NumPy's optimized backend."""
    # if a.dtype == np.float32:
    #    c = slb.sgemm(1.0, a, b)
    # elif a.dtype == np.float64:
    #    c = slb.dgemm(1.0, a, b)
    # else:
    # Native numpy matmul gets more performance than scipy blas!
    # (added later: thats because numpy uses an optimized blas implementation)
    if c is None:
        return a @ b
    else:
        return np.matmul(a, b, c)