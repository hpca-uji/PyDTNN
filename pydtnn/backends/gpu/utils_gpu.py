"""
PyDTNN Utilities
"""

# 
#  This file is part of Python Distributed Training of Neural Networks (PyDTNN)
# 
#  Copyright (C) 2021-22 Universitat Jaume I
# 
#  PyDTNN is free software: you can redistribute it and/or modify it under the
#  terms of the GNU General Public License as published by the Free Software
#  Foundation, either version 3 of the License, or (at your option) any later
#  version.
# 
#  This program is distributed in the hope that it will be useful, but WITHOUT
#  ANY WARRANTY; without even the implied warranty of MERCHANTABILITY
#  or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public
#  License for more details.
# 
#  You should have received a copy of the GNU General Public License along
#  with this program.  If not, see <https://www.gnu.org/licenses/>.
# 

try:
    # noinspection PyUnresolvedReferences
    from skcuda import cublas
except (ImportError, ModuleNotFoundError):
    pass

import numpy as np

def matmul_gpu(handle, trans_a, trans_b, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc, dtype):
    try:
        gemm = {np.float32: cublas.cublasSgemm,
                np.float64: cublas.cublasDgemm}[dtype]
    except KeyError:
        print("I cannot handle %s type!\n" % dtype.__name__)
    else:
        gemm(handle, trans_a, trans_b, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc)


def matvec_gpu(handle, trans_a, m, n, alpha, a, lda, b, ldb, beta, c, ldc, dtype):
    try:
        gemv = {np.float32: cublas.cublasSgemv,
                np.float64: cublas.cublasDgemv}[dtype]
    except KeyError:
        print("I cannot handle %s type!\n" % dtype.__name__)
    else:
        gemv(handle, trans_a, m, n, alpha, a, lda, b, ldb, beta, c, ldc)
