"""PyDTNN Utilities"""

import ctypes
import logging

import numpy as np

from pydtnn import Cublas_Handle_Type

__all__ = (
    "matmul_gpu",
    "matvec_gpu",
)

logger = logging.getLogger(__name__)

try:
    from pydtnn.libs import cublas  # type: ignore
except Exception:
    pass


def matmul_gpu(
    handle: Cublas_Handle_Type,
    trans_a: int | str,
    trans_b: int | str,
    m: int,
    n: int,
    k: int,
    alpha: float,
    a: ctypes.c_void_p,
    lda: int,
    b: ctypes.c_void_p,
    ldb: int,
    beta: float,
    c: ctypes.c_void_p,
    ldc: int,
    dtype: np.dtype,
) -> None:
    """
    Perform matrix-matrix multiplication on GPU using cuBLAS.

    Args:
        handle: cuBLAS handle.
        trans_a: Transposition state for matrix A.
        trans_b: Transposition state for matrix B.
        m: Number of rows of matrix A and C.
        n: Number of columns of matrix B and C.
        k: Number of columns of A and rows of B.
        alpha: Scalar multiplier for A*B.
        a: Pointer to matrix A.
        lda: Leading dimension of A.
        b: Pointer to matrix B.
        ldb: Leading dimension of B.
        beta: Scalar multiplier for C.
        c: Pointer to matrix C.
        ldc: Leading dimension of C.
        dtype: Data type of the matrices.
    """
    try:
        gemm = {np.dtype(np.float32): cublas.cublasSgemm, np.dtype(np.float64): cublas.cublasDgemm}[
            dtype
        ]
    except KeyError:
        logger.error("I cannot handle %s type!\n" % dtype.__name__)
    else:
        gemm(handle, trans_a, trans_b, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc)


def matvec_gpu(
    handle: int,
    trans_a: str | int,
    m: int,
    n: int,
    alpha: float,
    a: ctypes.c_void_p,
    lda: int,
    b: ctypes.c_void_p,
    ldb: int,
    beta: float,
    c: ctypes.c_void_p,
    ldc: int,
    dtype: np.dtype,
) -> None:
    """
    Perform matrix-vector multiplication on GPU using cuBLAS.

    Args:
        handle: cuBLAS handle.
        trans_a: Transposition operation for matrix A.
        m: Number of rows of matrix A.
        n: Number of columns of matrix A.
        alpha: Scalar multiplier for A*x.
        a: Pointer to matrix A.
        lda: Leading dimension of A.
        b: Pointer to vector x.
        ldb: Stride of vector x.
        beta: Scalar multiplier for y.
        c: Pointer to vector y.
        ldc: Stride of vector y.
        dtype: Data type of the matrix and vectors.
    """
    try:
        gemv = {np.dtype(np.float32): cublas.cublasSgemv, np.dtype(np.float64): cublas.cublasDgemv}[
            dtype
        ]
    except KeyError:
        logger.error("I cannot handle %s type!\n" % dtype.__name__)
    else:
        gemv(handle, trans_a, m, n, alpha, a, lda, b, ldb, beta, c, ldc)
