"""
PyDTNN Utilities
"""

try:
    from skcuda import cublas  #type: ignore
except Exception:
    pass

import numpy as np


def matmul_gpu(handle, trans_a, trans_b, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc, dtype):
    try:
        gemm = {np.dtype(np.float32): cublas.cublasSgemm,
                np.dtype(np.float64): cublas.cublasDgemm}[dtype]
    except KeyError:
        print("I cannot handle %s type!\n" % dtype.__name__)
    else:
        gemm(handle, trans_a, trans_b, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc)


def matvec_gpu(handle, trans_a, m, n, alpha, a, lda, b, ldb, beta, c, ldc, dtype):
    try:
        gemv = {np.dtype(np.float32): cublas.cublasSgemv,
                np.dtype(np.float64): cublas.cublasDgemv}[dtype]
    except KeyError:
        print("I cannot handle %s type!\n" % dtype.__name__)
    else:
        gemv(handle, trans_a, m, n, alpha, a, lda, b, ldb, beta, c, ldc)
