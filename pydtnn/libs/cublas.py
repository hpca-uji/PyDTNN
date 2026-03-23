# -*- coding: utf-8 -*-

"""
Raw ctypes wrappers of the cuBLAS library v7.0+ (libcublas.so.7.0)

For documentation see:
http://docs.nvidia.com/cuda/cublas
/usr/include/ cublas_api.h and cublas_v2.h
"""

import platform
import ctypes
import ctypes.util
import enum
from ctypes import *

### cuBLAS Library ###
libname = ctypes.util.find_library('cublas')
# TODO import name for windows/mac?
if platform.system() == 'Windows':
    libcublas = ctypes.windll.LoadLibrary(libname)
elif platform.system() == 'Linux':
    libcublas = ctypes.CDLL(libname, ctypes.RTLD_GLOBAL)
else:
    libcublas = ctypes.cdll.LoadLibrary(libname)


## cuBLAS Datatypes ##

# cublasStatus_t
class cublasStatus_t(enum.IntEnum):
    CUBLAS_STATUS_SUCCESS = 0
    CUBLAS_STATUS_NOT_INITIALIZED = 1
    CUBLAS_STATUS_ALLOC_FAILED = 3
    CUBLAS_STATUS_INVALID_VALUE = 7
    CUBLAS_STATUS_ARCH_MISMATCH = 8
    CUBLAS_STATUS_MAPPING_ERROR = 11
    CUBLAS_STATUS_EXECUTION_FAILED = 13
    CUBLAS_STATUS_INTERNAL_ERROR = 14
    CUBLAS_STATUS_NOT_SUPPORTED = 15

# cublasFillMode_t


class cublasFillMode_t(enum.IntEnum):
    CUBLAS_FILL_MODE_LOWER = 0
    CUBLAS_FILL_MODE_UPPER = 1


c_cublasFillMode_t = c_int

# cublasDiagType_t


class cublasDiagType_t(enum.IntEnum):
    CUBLAS_DIAG_NON_UNIT = 0
    CUBLAS_DIAG_UNIT = 1


c_cublasDiagType_t = c_int

# cublasSideMode_t


class cublasSideMode_t(enum.IntEnum):
    CUBLAS_SIDE_LEFT = 0
    CUBLAS_SIDE_RIGHT = 1


c_cublasSideMode_t = c_int

# cublasOperation_t


class cublasOperation_t(enum.IntEnum):
    CUBLAS_OP_N = 0
    CUBLAS_OP_T = 1
    CUBLAS_OP_C = 2


c_cublasOperation_t = c_int

# cublasPointerMode_t


class cublasPointerMode_t(enum.IntEnum):
    CUBLAS_POINTER_MODE_HOST = 0
    CUBLAS_POINTER_MODE_DEVICE = 1


c_cublasPointerMode_t = c_int

# cublasAtomicsMode_t


class cublasAtomicsMode_t(enum.IntEnum):
    CUBLAS_ATOMICS_NOT_ALLOWED = 0
    CUBLAS_ATOMICS_ALLOWED = 1


c_cublasAtomicsMode_t = c_int

# /* Opaque structure holding CUBLAS library context */
# struct cublasContext;
# typedef struct cublasContext *cublasHandle_t;


class _opaque(ctypes.Structure):
    pass


cublasHandle_t = POINTER(_opaque)
cublasHandle_t.__name__ = 'cublasHandle_t'

memory_pointer = ctypes.c_void_p
array_pointer = ctypes.c_void_p
result_pointer = ctypes.c_void_p
scalar_pointer = ctypes.c_void_p
param_pointer = ctypes.c_void_p

## cuBLAS Helper Functions ##

# cublasStatus_t cublasCreate(cublasHandle_t *handle)
cublasCreate = libcublas.cublasCreate_v2
cublasCreate.restype = cublasStatus_t
cublasCreate.argtypes = [POINTER(cublasHandle_t)]

# cublasStatus_t cublasDestroy(cublasHandle_t handle)
cublasDestroy = libcublas.cublasDestroy_v2
cublasDestroy.restype = cublasStatus_t
cublasDestroy.argtypes = [cublasHandle_t]

# cublasStatus_t cublasGetVersion(cublasHandle_t handle, int *version)
cublasGetVersion = libcublas.cublasGetVersion_v2
cublasGetVersion.restype = cublasStatus_t
cublasGetVersion.argtypes = [cublasHandle_t, POINTER(c_int)]

# cublasStatus_t = cublasSetStream(cublasHandle_t handle, cudaStream_t streamId)
# cublasStatus_t = cublasGetStream(cublasHandle_t handle, cudaStream_t *streamId)

# cublasStatus_t cublasGetPointerMode(cublasHandle_t handle, cublasPointerMode_t *mode)
cublasGetPointerMode = libcublas.cublasGetPointerMode_v2
cublasGetPointerMode.restype = cublasStatus_t
cublasGetPointerMode.argtypes = [cublasHandle_t, POINTER(c_cublasPointerMode_t)]

# cublasStatus_t cublasSetPointerMode(cublasHandle_t handle, cublasPointerMode_t mode)
cublasSetPointerMode = libcublas.cublasSetPointerMode_v2
cublasSetPointerMode.restype = cublasStatus_t
cublasSetPointerMode.argtypes = [cublasHandle_t, c_cublasPointerMode_t]

# cublasStatus_t cublasSetVector(int n, int elemSize,
#                                const void *x, int incx, void *y, int incy)
# cublasStatus_t cublasGetVector(int n, int elemSize,
#                                const void *x, int incx, void *y, int incy)
cublasSetVector = libcublas.cublasSetVector
cublasGetVector = libcublas.cublasGetVector
for funct in [cublasSetVector, cublasGetVector]:
    funct.restype = cublasStatus_t
    funct.argtypes = [c_int, c_int,  # n, elemSize
                      memory_pointer, c_int,  # *x, incx
                      memory_pointer, c_int  # *y, incy
                      ]

# cublasStatus_t cublasSetMatrix(int rows, int cols, int elemSize,
#                                const void *A, int lda, void *B, int ldb)
# cublasStatus_t cublasGetMatrix(int rows, int cols, int elemSize,
#                                const void *A, int lda, void *B, int ldb)
cublasSetMatrix = libcublas.cublasSetMatrix
cublasGetMatrix = libcublas.cublasGetMatrix
for funct in [cublasSetMatrix, cublasGetMatrix]:
    funct.restype = cublasStatus_t
    funct.argtypes = [c_int, c_int, c_int,  # rows, cols, elemSize
                      memory_pointer, c_int,  # *A, incx
                      memory_pointer, c_int  # *B, incy
                      ]

# cublasStatus_t cublasSetVectorAsync(int n, int elemSize, const void *hostPtr, int incx,
#                                     void *devicePtr, int incy, cudaStream_t stream)

# cublasStatus_t cublasGetVectorAsync(int n, int elemSize, const void *devicePtr, int incx,
#                                     void *hostPtr, int incy, cudaStream_t stream)

# cublasStatus_t cublasSetMatrixAsync(int rows, int cols, int elemSize, const void *A,
#                                     int lda, void *B, int ldb, cudaStream_t stream)

# cublasStatus_t cublasGetMatrixAsync(int rows, int cols, int elemSize, const void *A,
#                                     int lda, void *B, int ldb, cudaStream_t stream)


# cublasStatus_t cublasSetAtomicsMode(cublasHandlet handle, cublasAtomicsMode_t mode)
cublasGetAtomicsMode = libcublas.cublasGetAtomicsMode
cublasGetAtomicsMode.restype = cublasStatus_t
cublasGetAtomicsMode.argtypes = [cublasHandle_t, POINTER(c_cublasAtomicsMode_t)]

# cublasStatus_t cublasSetAtomicsMode(cublasHandlet handle, cublasAtomicsMode_t mode)
cublasSetAtomicsMode = libcublas.cublasSetAtomicsMode
cublasSetAtomicsMode.restype = cublasStatus_t
cublasSetAtomicsMode.argtypes = [cublasHandle_t, c_cublasAtomicsMode_t]


## cuBLAS Level-1 Functions ##

# cublasStatus_t cublasIsamax(cublasHandle_t handle, int n,
#                             const float *x, int incx, int *result)
# cublasStatus_t cublasIdamax(cublasHandle_t handle, int n,
#                             const double *x, int incx, int *result)
# cublasStatus_t cublasIcamax(cublasHandle_t handle, int n,
#                             const cuComplex *x, int incx, int *result)
# cublasStatus_t cublasIzamax(cublasHandle_t handle, int n,
#                             const cuDoubleComplex *x, int incx, int *result)
cublasIsamax = libcublas.cublasIsamax_v2
cublasIdamax = libcublas.cublasIdamax_v2
cublasIcamax = libcublas.cublasIcamax_v2
cublasIzamax = libcublas.cublasIzamax_v2
for funct in [cublasIsamax, cublasIdamax, cublasIcamax, cublasIzamax]:
    funct.restype = cublasStatus_t
    funct.argtypes = [cublasHandle_t,
                      c_int,  # n
                      memory_pointer, c_int,  # *x, incx
                      POINTER(c_int)  # *result
                      ]

# cublasStatus_t cublasIsamin(cublasHandle_t handle, int n,
#                             const float *x, int incx, int *result)
# cublasStatus_t cublasIdamin(cublasHandle_t handle, int n,
#                             const double *x, int incx, int *result)
# cublasStatus_t cublasIcamin(cublasHandle_t handle, int n,
#                             const cuComplex *x, int incx, int *result)
# cublasStatus_t cublasIzamin(cublasHandle_t handle, int n,
#                             const cuDoubleComplex *x, int incx, int *result)
cublasIsamin = libcublas.cublasIsamin_v2
cublasIdamin = libcublas.cublasIdamin_v2
cublasIcamin = libcublas.cublasIcamin_v2
cublasIzamin = libcublas.cublasIzamin_v2
for funct in [cublasIsamin, cublasIdamin, cublasIcamin, cublasIzamin]:
    funct.restype = cublasStatus_t
    funct.argtypes = [cublasHandle_t,
                      c_int,  # n
                      memory_pointer, c_int,  # *x, incx
                      POINTER(c_int)  # *result
                      ]

# cublasStatus_t  cublasSasum(cublasHandle_t handle, int n,
#                             const float           *x, int incx, float  *result)
# cublasStatus_t  cublasDasum(cublasHandle_t handle, int n,
#                             const double          *x, int incx, double *result)
# cublasStatus_t cublasScasum(cublasHandle_t handle, int n,
#                             const cuComplex       *x, int incx, float  *result)
# cublasStatus_t cublasDzasum(cublasHandle_t handle, int n,
#                             const cuDoubleComplex *x, int incx, double *result)
cublasSasum = libcublas.cublasSasum_v2
cublasDasum = libcublas.cublasDasum_v2
cublasScasum = libcublas.cublasScasum_v2
cublasDzasum = libcublas.cublasDzasum_v2
for (funct, result_type) in [(cublasSasum, c_float), (cublasDasum, c_double),
                             (cublasScasum, c_float), (cublasDzasum, c_double)]:
    funct.restype = cublasStatus_t
    funct.argtypes = [cublasHandle_t,
                      c_int,  # n
                      memory_pointer, c_int,  # *x, incx
                      POINTER(result_type)  # *result
                      ]

# cublasStatus_t cublasSaxpy(cublasHandle_t handle, int n,
#                            const float           *alpha,
#                            const float           *x, int incx,
#                            float                 *y, int incy)
# cublasStatus_t cublasDaxpy(cublasHandle_t handle, int n,
#                            const double          *alpha,
#                            const double          *x, int incx,
#                            double                *y, int incy)
# cublasStatus_t cublasCaxpy(cublasHandle_t handle, int n,
#                            const cuComplex       *alpha,
#                            const cuComplex       *x, int incx,
#                            cuComplex             *y, int incy)
# cublasStatus_t cublasZaxpy(cublasHandle_t handle, int n,
#                            const cuDoubleComplex *alpha,
#                            const cuDoubleComplex *x, int incx,
#                            cuDoubleComplex       *y, int incy)
cublasSaxpy = libcublas.cublasSaxpy_v2
cublasDaxpy = libcublas.cublasDaxpy_v2
cublasCaxpy = libcublas.cublasCaxpy_v2
cublasZaxpy = libcublas.cublasZaxpy_v2
for funct in [cublasSaxpy, cublasDaxpy, cublasCaxpy, cublasZaxpy]:
    funct.restype = cublasStatus_t
    funct.argtypes = [cublasHandle_t,
                      c_int,  # n
                      scalar_pointer,  # *alpha
                      memory_pointer, c_int,  # *x, incx
                      memory_pointer, c_int  # *y, incy
                      ]

# cublasStatus_t cublasScopy(cublasHandle_t handle, int n,
#                            const float           *x, int incx,
#                            float                 *y, int incy)
# cublasStatus_t cublasDcopy(cublasHandle_t handle, int n,
#                            const double          *x, int incx,
#                            double                *y, int incy)
# cublasStatus_t cublasCcopy(cublasHandle_t handle, int n,
#                            const cuComplex       *x, int incx,
#                            cuComplex             *y, int incy)
# cublasStatus_t cublasZcopy(cublasHandle_t handle, int n,
#                            const cuDoubleComplex *x, int incx,
#                            cuDoubleComplex       *y, int incy)
cublasScopy = libcublas.cublasScopy_v2
cublasDcopy = libcublas.cublasDcopy_v2
cublasCcopy = libcublas.cublasCcopy_v2
cublasZcopy = libcublas.cublasZcopy_v2
for funct in [cublasScopy, cublasDcopy, cublasCcopy, cublasZcopy]:
    funct.restype = cublasStatus_t
    funct.argtypes = [cublasHandle_t,
                      c_int,  # n
                      memory_pointer, c_int,  # *x, incx
                      memory_pointer, c_int  # *y, incy
                      ]

# cublasStatus_t cublasSdot (cublasHandle_t handle, int n,
#                            const float           *x, int incx,
#                            const float           *y, int incy,
#                            float           *result)
# cublasStatus_t cublasDdot (cublasHandle_t handle, int n,
#                            const double          *x, int incx,
#                            const double          *y, int incy,
#                            double          *result)
# cublasStatus_t cublasCdotu(cublasHandle_t handle, int n,
#                            const cuComplex       *x, int incx,
#                            const cuComplex       *y, int incy,
#                            cuComplex       *result)
# cublasStatus_t cublasCdotc(cublasHandle_t handle, int n,
#                            const cuComplex       *x, int incx,
#                            const cuComplex       *y, int incy,
#                            cuComplex       *result)
# cublasStatus_t cublasZdotu(cublasHandle_t handle, int n,
#                            const cuDoubleComplex *x, int incx,
#                            const cuDoubleComplex *y, int incy,
#                            cuDoubleComplex *result)
# cublasStatus_t cublasZdotc(cublasHandle_t handle, int n,
#                            const cuDoubleComplex *x, int incx,
#                            const cuDoubleComplex *y, int incy,
#                            cuDoubleComplex       *result)
cublasSdot = libcublas.cublasSdot_v2
cublasDdot = libcublas.cublasDdot_v2
cublasCdotu = libcublas.cublasCdotu_v2
cublasCdotc = libcublas.cublasCdotc_v2
cublasZdotu = libcublas.cublasZdotu_v2
cublasZdotc = libcublas.cublasZdotc_v2
for funct in [cublasSdot, cublasDdot,
              cublasCdotu, cublasCdotc,
              cublasZdotu, cublasZdotc]:
    funct.restype = cublasStatus_t
    funct.argtypes = [cublasHandle_t,
                      c_int,  # n
                      memory_pointer, c_int,  # *x, incx
                      memory_pointer, c_int,  # *y, incy
                      result_pointer  # *result
                      ]

# cublasStatus_t  cublasSnrm2(cublasHandle_t handle, int n,
#                             const float           *x, int incx, float  *result)
# cublasStatus_t  cublasDnrm2(cublasHandle_t handle, int n,
#                             const double          *x, int incx, double *result)
# cublasStatus_t cublasScnrm2(cublasHandle_t handle, int n,
#                             const cuComplex       *x, int incx, float  *result)
# cublasStatus_t cublasDznrm2(cublasHandle_t handle, int n,
#                             const cuDoubleComplex *x, int incx, double *result)
cublasSnrm2 = libcublas.cublasSnrm2_v2
cublasDnrm2 = libcublas.cublasDnrm2_v2
cublasScnrm2 = libcublas.cublasScnrm2_v2
cublasDznrm2 = libcublas.cublasDznrm2_v2
for (funct, result_type) in [(cublasSnrm2, c_float), (cublasDnrm2, c_double),
                             (cublasScnrm2, c_float), (cublasDznrm2, c_double)]:
    funct.restype = cublasStatus_t
    funct.argtypes = [cublasHandle_t,
                      c_int,  # n
                      memory_pointer, c_int,  # *x, incx
                      POINTER(result_type)  # *result
                      ]

# cublasStatus_t  cublasSrot(cublasHandle_t handle, int n,
#                            float           *x, int incx,
#                            float           *y, int incy,
#                            const float  *c, const float           *s)
# cublasStatus_t  cublasDrot(cublasHandle_t handle, int n,
#                            double          *x, int incx,
#                            double          *y, int incy,
#                            const double *c, const double          *s)
# cublasStatus_t  cublasCrot(cublasHandle_t handle, int n,
#                            cuComplex       *x, int incx,
#                            cuComplex       *y, int incy,
#                            const float  *c, const cuComplex       *s)
# cublasStatus_t cublasCsrot(cublasHandle_t handle, int n,
#                            cuComplex       *x, int incx,
#                            cuComplex       *y, int incy,
#                            const float  *c, const float           *s)
# cublasStatus_t  cublasZrot(cublasHandle_t handle, int n,
#                            cuDoubleComplex *x, int incx,
#                            cuDoubleComplex *y, int incy,
#                            const double *c, const cuDoubleComplex *s)
# cublasStatus_t cublasZdrot(cublasHandle_t handle, int n,
#                            cuDoubleComplex *x, int incx,
#                            cuDoubleComplex *y, int incy,
#                            const double *c, const double          *s)
cublasSrot = libcublas.cublasSrot_v2
cublasDrot = libcublas.cublasDrot_v2
cublasCrot = libcublas.cublasCrot_v2
cublasCsrot = libcublas.cublasCsrot_v2
cublasZrot = libcublas.cublasZrot_v2
cublasZdrot = libcublas.cublasZdrot_v2
for funct in [cublasSrot, cublasDrot,
              cublasCrot, cublasCsrot,
              cublasZrot, cublasZdrot]:
    funct.restype = cublasStatus_t
    funct.argtypes = [cublasHandle_t,
                      c_int,  # *n
                      memory_pointer, c_int,  # *x, incx
                      memory_pointer, c_int,  # *y, incy
                      scalar_pointer,  # *c
                      scalar_pointer  # *s
                      ]

# cublasStatus_t cublasSrotg(cublasHandle_t handle,
#                            float           *a, float           *b,
#                            float  *c, float           *s)
# cublasStatus_t cublasDrotg(cublasHandle_t handle,
#                            double          *a, double          *b,
#                            double *c, double          *s)
# cublasStatus_t cublasCrotg(cublasHandle_t handle,
#                            cuComplex       *a, cuComplex       *b,
#                            float  *c, cuComplex       *s)
# cublasStatus_t cublasZrotg(cublasHandle_t handle,
#                            cuDoubleComplex *a, cuDoubleComplex *b,
#                            double *c, cuDoubleComplex *s)
cublasSrotg = libcublas.cublasSrotg_v2
cublasDrotg = libcublas.cublasDrotg_v2
cublasCrotg = libcublas.cublasCrotg_v2
cublasZrotg = libcublas.cublasZrotg_v2
for funct in [cublasSrotg, cublasDrotg, cublasCrotg, cublasZrotg]:
    funct.restype = cublasStatus_t
    funct.argtypes = [cublasHandle_t,
                      scalar_pointer,  # *a
                      scalar_pointer,  # *b
                      result_pointer,  # *c
                      result_pointer  # *s
                      ]

# cublasStatus_t cublasSrotm(cublasHandle_t handle, int n, float  *x, int incx,
#                            float  *y, int incy, const float*  param)
# cublasStatus_t cublasDrotm(cublasHandle_t handle, int n, double *x, int incx,
#                            double *y, int incy, const double* param)
cublasSrotm = libcublas.cublasSrotm_v2
cublasDrotm = libcublas.cublasDrotm_v2
for funct in [cublasSrotm, cublasDrotm]:
    funct.restype = cublasStatus_t
    funct.argtypes = [cublasHandle_t,
                      c_int,  # n
                      memory_pointer, c_int,  # *x, incx
                      memory_pointer, c_int,  # *y, incy
                      param_pointer  # *param
                      ]

# cublasStatus_t cublasSrotmg(cublasHandle_t handle, float  *d1, float  *d2,
#                             float  *x1, const float  *y1, float  *param)
# cublasStatus_t cublasDrotmg(cublasHandle_t handle, double *d1, double *d2,
#                             double *x1, const double *y1, double *param)
cublasSrotmg = libcublas.cublasSrotmg_v2
cublasDrotmg = libcublas.cublasDrotmg_v2
for funct in [cublasSrotmg, cublasDrotmg]:
    funct.restype = cublasStatus_t
    funct.argtypes = [cublasHandle_t,
                      scalar_pointer,  # *d1
                      scalar_pointer,  # *d2
                      scalar_pointer,  # *x1
                      scalar_pointer,  # *y1
                      param_pointer  # *param
                      ]

# cublasStatus_t  cublasSscal(cublasHandle_t handle, int n,
#                             const float           *alpha,
#                             float           *x, int incx)
# cublasStatus_t  cublasDscal(cublasHandle_t handle, int n,
#                             const double          *alpha,
#                             double          *x, int incx)
# cublasStatus_t  cublasCscal(cublasHandle_t handle, int n,
#                             const cuComplex       *alpha,
#                             cuComplex       *x, int incx)
# cublasStatus_t cublasCsscal(cublasHandle_t handle, int n,
#                             const float           *alpha,
#                             cuComplex       *x, int incx)
# cublasStatus_t  cublasZscal(cublasHandle_t handle, int n,
#                             const cuDoubleComplex *alpha,
#                             cuDoubleComplex *x, int incx)
# cublasStatus_t cublasZdscal(cublasHandle_t handle, int n,
#                             const double          *alpha,
#                             cuDoubleComplex *x, int incx)
cublasSscal = libcublas.cublasSscal_v2
cublasDscal = libcublas.cublasDscal_v2
cublasCscal = libcublas.cublasCscal_v2
cublasCsscal = libcublas.cublasCsscal_v2
cublasZscal = libcublas.cublasZscal_v2
cublasZdscal = libcublas.cublasZdscal_v2
for funct in [cublasSscal, cublasDscal,
              cublasCscal, cublasCsscal,
              cublasZscal, cublasZdscal]:
    funct.restype = cublasStatus_t
    funct.argtypes = [cublasHandle_t,
                      c_int,  # n
                      scalar_pointer,  # *alpha
                      memory_pointer, c_int  # *x, incx
                      ]

# cublasStatus_t cublasSswap(cublasHandle_t handle, int n, float           *x,
#                            int incx, float           *y, int incy)
# cublasStatus_t cublasDswap(cublasHandle_t handle, int n, double          *x,
#                            int incx, double          *y, int incy)
# cublasStatus_t cublasCswap(cublasHandle_t handle, int n, cuComplex       *x,
#                            int incx, cuComplex       *y, int incy)
# cublasStatus_t cublasZswap(cublasHandle_t handle, int n, cuDoubleComplex *x,
#                            int incx, cuDoubleComplex *y, int incy)
cublasSswap = libcublas.cublasSswap_v2
cublasDswap = libcublas.cublasDswap_v2
cublasCswap = libcublas.cublasCswap_v2
cublasZswap = libcublas.cublasZswap_v2
for funct in [cublasSswap, cublasDswap, cublasCswap, cublasZswap]:
    funct.restype = cublasStatus_t
    funct.argtypes = [cublasHandle_t,
                      c_int,  # n
                      memory_pointer, c_int,  # *x, incx
                      memory_pointer, c_int  # *y, incy
                      ]

## cuBLAS Level-2 Functions ##

# cublasStatus_t cublasSgbmv(cublasHandle_t handle, cublasOperation_t trans,
#                            int m, int n, int kl, int ku,
#                            const float           *alpha,
#                            const float           *A, int lda,
#                            const float           *x, int incx,
#                            const float           *beta,
#                            float           *y, int incy)
# cublasStatus_t cublasDgbmv(cublasHandle_t handle, cublasOperation_t trans,
#                            int m, int n, int kl, int ku,
#                            const double          *alpha,
#                            const double          *A, int lda,
#                            const double          *x, int incx,
#                            const double          *beta,
#                            double          *y, int incy)
# cublasStatus_t cublasCgbmv(cublasHandle_t handle, cublasOperation_t trans,
#                            int m, int n, int kl, int ku,
#                            const cuComplex       *alpha,
#                            const cuComplex       *A, int lda,
#                            const cuComplex       *x, int incx,
#                            const cuComplex       *beta,
#                            cuComplex       *y, int incy)
# cublasStatus_t cublasZgbmv(cublasHandle_t handle, cublasOperation_t trans,
#                            int m, int n, int kl, int ku,
#                            const cuDoubleComplex *alpha,
#                            const cuDoubleComplex *A, int lda,
#                            const cuDoubleComplex *x, int incx,
#                            const cuDoubleComplex *beta,
#                            cuDoubleComplex *y, int incy)
cublasSgbmv = libcublas.cublasSgbmv_v2
cublasDgbmv = libcublas.cublasDgbmv_v2
cublasCgbmv = libcublas.cublasCgbmv_v2
cublasZgbmv = libcublas.cublasZgbmv_v2
for funct in [cublasSgbmv, cublasDgbmv, cublasCgbmv, cublasZgbmv]:
    funct.restype = cublasStatus_t
    funct.argtypes = [cublasHandle_t,
                      c_cublasOperation_t,  # trans
                      c_int, c_int,  # m, n
                      c_int, c_int,  # kl, ku
                      scalar_pointer,  # *alpha
                      memory_pointer, c_int,  # *A, lda
                      memory_pointer, c_int,  # *x, incx
                      scalar_pointer,  # *beta
                      memory_pointer, c_int,  # *y, incy
                      ]

# cublasStatus_t cublasSgemv(cublasHandle_t handle, cublasOperation_t trans,
#                            int m, int n,
#                            const float           *alpha,
#                            const float           *A, int lda,
#                            const float           *x, int incx,
#                            const float           *beta,
#                            float           *y, int incy)
# cublasStatus_t cublasDgemv(cublasHandle_t handle, cublasOperation_t trans,
#                            int m, int n,
#                            const double          *alpha,
#                            const double          *A, int lda,
#                            const double          *x, int incx,
#                            const double          *beta,
#                            double          *y, int incy)
# cublasStatus_t cublasCgemv(cublasHandle_t handle, cublasOperation_t trans,
#                            int m, int n,
#                            const cuComplex       *alpha,
#                            const cuComplex       *A, int lda,
#                            const cuComplex       *x, int incx,
#                            const cuComplex       *beta,
#                            cuComplex       *y, int incy)
# cublasStatus_t cublasZgemv(cublasHandle_t handle, cublasOperation_t trans,
#                            int m, int n,
#                            const cuDoubleComplex *alpha,
#                            const cuDoubleComplex *A, int lda,
#                            const cuDoubleComplex *x, int incx,
#                            const cuDoubleComplex *beta,
#                            cuDoubleComplex *y, int incy)
cublasSgemv = libcublas.cublasSgemv_v2
cublasDgemv = libcublas.cublasDgemv_v2
cublasCgemv = libcublas.cublasCgemv_v2
cublasZgemv = libcublas.cublasZgemv_v2
for funct in [cublasSgemv, cublasDgemv, cublasCgemv, cublasZgemv]:
    funct.restype = cublasStatus_t
    funct.argtypes = [cublasHandle_t,
                      c_cublasOperation_t,  # trans
                      c_int, c_int,  # m, n
                      scalar_pointer,  # *alpha
                      memory_pointer, c_int,  # *A, lda
                      memory_pointer, c_int,  # *x, incx
                      scalar_pointer,  # *beta
                      memory_pointer, c_int,  # *y, incy
                      ]

# cublasStatus_t  cublasSger(cublasHandle_t handle, int m, int n,
#                            const float           *alpha,
#                            const float           *x, int incx,
#                            const float           *y, int incy,
#                            float           *A, int lda)
# cublasStatus_t  cublasDger(cublasHandle_t handle, int m, int n,
#                            const double          *alpha,
#                            const double          *x, int incx,
#                            const double          *y, int incy,
#                            double          *A, int lda)
# cublasStatus_t cublasCgeru(cublasHandle_t handle, int m, int n,
#                            const cuComplex       *alpha,
#                            const cuComplex       *x, int incx,
#                            const cuComplex       *y, int incy,
#                            cuComplex       *A, int lda)
# cublasStatus_t cublasCgerc(cublasHandle_t handle, int m, int n,
#                            const cuComplex       *alpha,
#                            const cuComplex       *x, int incx,
#                            const cuComplex       *y, int incy,
#                            cuComplex       *A, int lda)
# cublasStatus_t cublasZgeru(cublasHandle_t handle, int m, int n,
#                            const cuDoubleComplex *alpha,
#                            const cuDoubleComplex *x, int incx,
#                            const cuDoubleComplex *y, int incy,
#                            cuDoubleComplex *A, int lda)
# cublasStatus_t cublasZgerc(cublasHandle_t handle, int m, int n,
#                            const cuDoubleComplex *alpha,
#                            const cuDoubleComplex *x, int incx,
#                            const cuDoubleComplex *y, int incy,
#                            cuDoubleComplex *A, int lda)
cublasSger = libcublas.cublasSger_v2
cublasDger = libcublas.cublasDger_v2
cublasCgeru = libcublas.cublasCgeru_v2
cublasCgerc = libcublas.cublasCgerc_v2
cublasZgeru = libcublas.cublasZgeru_v2
cublasZgerc = libcublas.cublasZgerc_v2
for funct in [cublasSger, cublasDger,
              cublasCgeru, cublasCgerc,
              cublasZgeru, cublasZgerc]:
    funct.restype = cublasStatus_t
    funct.argtypes = [cublasHandle_t,
                      c_int, c_int,  # m, n
                      scalar_pointer,  # *alpha
                      memory_pointer, c_int,  # *x, incx
                      memory_pointer, c_int,  # *y, incy
                      memory_pointer, c_int  # *A, lda
                      ]

# cublasStatus_t cublasSsbmv(cublasHandle_t handle, cublasFillMode_t uplo,
#                            int n, int k, const float  *alpha,
#                            const float  *A, int lda,
#                            const float  *x, int incx,
#                            const float  *beta, float *y, int incy)
# cublasStatus_t cublasDsbmv(cublasHandle_t handle, cublasFillMode_t uplo,
#                            int n, int k, const double *alpha,
#                            const double *A, int lda,
#                            const double *x, int incx,
#                            const double *beta, double *y, int incy)
cublasSsbmv = libcublas.cublasSsbmv_v2
cublasDsbmv = libcublas.cublasDsbmv_v2
for funct in [cublasSsbmv, cublasDsbmv]:
    funct.restype = cublasStatus_t
    funct.argtypes = [cublasHandle_t,
                      c_cublasFillMode_t,  # uplo
                      c_int, c_int,  # n, k
                      scalar_pointer,  # *alpha
                      memory_pointer, c_int,  # *A, lda
                      memory_pointer, c_int,  # *x, incx
                      scalar_pointer,  # *beta
                      memory_pointer, c_int  # *y, incy
                      ]

# cublasStatus_t cublasSspmv(cublasHandle_t handle, cublasFillMode_t uplo,
#                            int n, const float  *alpha, const float  *AP,
#                            const float  *x, int incx, const float  *beta,
#                            float  *y, int incy)
# cublasStatus_t cublasDspmv(cublasHandle_t handle, cublasFillMode_t uplo,
#                            int n, const double *alpha, const double *AP,
#                            const double *x, int incx, const double *beta,
#                            double *y, int incy)
cublasSspmv = libcublas.cublasSspmv_v2
cublasDspmv = libcublas.cublasDspmv_v2
for funct in [cublasSspmv, cublasDspmv]:
    funct.restype = cublasStatus_t
    funct.argtypes = [cublasHandle_t,
                      c_cublasFillMode_t,  # uplo
                      c_int,  # n
                      scalar_pointer,  # *alpha
                      memory_pointer,  # *AP
                      memory_pointer, c_int,  # *x, incx
                      scalar_pointer,  # *beta
                      memory_pointer, c_int  # *y, incy
                      ]

# cublasStatus_t cublasSspr(cublasHandle_t handle, cublasFillMode_t uplo,
#                           int n, const float  *alpha,
#                           const float  *x, int incx, float  *AP)
# cublasStatus_t cublasDspr(cublasHandle_t handle, cublasFillMode_t uplo,
#                           int n, const double *alpha,
#                           const double *x, int incx, double *AP)
cublasSspr = libcublas.cublasSspr_v2
cublasDspr = libcublas.cublasDspr_v2
for funct in [cublasSspr, cublasDspr]:
    funct.restype = cublasStatus_t
    funct.argtypes = [cublasHandle_t,
                      c_cublasFillMode_t,  # uplo
                      c_int,  # n
                      scalar_pointer,  # *alpha
                      memory_pointer, c_int,  # *x, incx
                      memory_pointer  # *AP
                      ]

# cublasStatus_t cublasSspr2(cublasHandle_t handle, cublasFillMode_t uplo,
#                            int n, const float  *alpha,
#                            const float  *x, int incx,
#                            const float  *y, int incy, float  *AP)
# cublasStatus_t cublasDspr2(cublasHandle_t handle, cublasFillMode_t uplo,
#                            int n, const double *alpha,
#                            const double *x, int incx,
#                            const double *y, int incy, double *AP)
cublasSspr2 = libcublas.cublasSspr2_v2
cublasDspr2 = libcublas.cublasDspr2_v2
for funct in [cublasSspr2, cublasDspr2]:
    funct.restype = cublasStatus_t
    funct.argtypes = [cublasHandle_t,
                      c_cublasFillMode_t,  # uplo
                      c_int,  # n
                      scalar_pointer,  # *alpha
                      memory_pointer, c_int,  # *x, incx
                      memory_pointer, c_int,  # *y, incy
                      memory_pointer  # *AP
                      ]

# cublasStatus_t cublasSsymv(cublasHandle_t handle, cublasFillMode_t uplo,
#                            int n, const float           *alpha,
#                            const float           *A, int lda,
#                            const float           *x, int incx, const float           *beta,
#                            float           *y, int incy)
# cublasStatus_t cublasDsymv(cublasHandle_t handle, cublasFillMode_t uplo,
#                            int n, const double          *alpha,
#                            const double          *A, int lda,
#                            const double          *x, int incx, const double          *beta,
#                            double          *y, int incy)
# cublasStatus_t cublasCsymv(cublasHandle_t handle, cublasFillMode_t uplo,
#                            int n, const cuComplex       *alpha, /* host or device pointer */
#                            const cuComplex       *A, int lda,
#                            const cuComplex       *x, int incx, const cuComplex       *beta,
#                            cuComplex       *y, int incy)
# cublasStatus_t cublasZsymv(cublasHandle_t handle, cublasFillMode_t uplo,
#                            int n, const cuDoubleComplex *alpha,
#                            const cuDoubleComplex *A, int lda,
#                            const cuDoubleComplex *x, int incx, const cuDoubleComplex *beta,
#                            cuDoubleComplex *y, int incy)
cublasSsymv = libcublas.cublasSsymv_v2
cublasDsymv = libcublas.cublasDsymv_v2
cublasCsymv = libcublas.cublasCsymv_v2
cublasZsymv = libcublas.cublasZsymv_v2
for funct in [cublasSsymv, cublasDsymv, cublasCsymv, cublasZsymv]:
    funct.restype = cublasStatus_t
    funct.argtypes = [cublasHandle_t,
                      c_cublasFillMode_t,  # uplo
                      c_int,  # n
                      scalar_pointer,  # *alpha
                      memory_pointer, c_int,  # *A, lda
                      memory_pointer, c_int,  # *x, incx
                      scalar_pointer,  # *beta
                      memory_pointer, c_int  # *y, incy
                      ]

# cublasStatus_t cublasSsyr(cublasHandle_t handle, cublasFillMode_t uplo,
#                           int n, const float           *alpha,
#                           const float           *x, int incx, float           *A, int lda)
# cublasStatus_t cublasDsyr(cublasHandle_t handle, cublasFillMode_t uplo,
#                           int n, const double          *alpha,
#                           const double          *x, int incx, double          *A, int lda)
# cublasStatus_t cublasCsyr(cublasHandle_t handle, cublasFillMode_t uplo,
#                           int n, const cuComplex       *alpha,
#                           const cuComplex       *x, int incx, cuComplex       *A, int lda)
# cublasStatus_t cublasZsyr(cublasHandle_t handle, cublasFillMode_t uplo,
#                           int n, const cuDoubleComplex *alpha,
#                           const cuDoubleComplex *x, int incx, cuDoubleComplex *A, int lda)
cublasSsyr = libcublas.cublasSsyr_v2
cublasDsyr = libcublas.cublasDsyr_v2
cublasCsyr = libcublas.cublasCsyr_v2
cublasZsyr = libcublas.cublasZsyr_v2
for funct in [cublasSsyr, cublasDsyr, cublasCsyr, cublasZsyr]:
    funct.restype = cublasStatus_t
    funct.argtypes = [cublasHandle_t,
                      c_cublasFillMode_t,  # uplo
                      c_int,  # n
                      scalar_pointer,  # *alpha
                      memory_pointer, c_int,  # *x, incx
                      memory_pointer, c_int  # *A, lda
                      ]

# cublasStatus_t cublasSsyr2(cublasHandle_t handle, cublasFillMode_t uplo, int n,
#                            const float           *alpha, const float           *x, int incx,
#                            const float           *y, int incy, float           *A, int lda
# cublasStatus_t cublasDsyr2(cublasHandle_t handle, cublasFillMode_t uplo, int n,
#                            const double          *alpha, const double          *x, int incx,
#                            const double          *y, int incy, double          *A, int lda
# cublasStatus_t cublasCsyr2(cublasHandle_t handle, cublasFillMode_t uplo, int n,
#                            const cuComplex       *alpha, const cuComplex       *x, int incx,
#                            const cuComplex       *y, int incy, cuComplex       *A, int lda
# cublasStatus_t cublasZsyr2(cublasHandle_t handle, cublasFillMode_t uplo, int n,
#                            const cuDoubleComplex *alpha, const cuDoubleComplex *x, int incx,
#                            const cuDoubleComplex *y, int incy, cuDoubleComplex *A, int lda
cublasSsyr2 = libcublas.cublasSsyr2_v2
cublasDsyr2 = libcublas.cublasDsyr2_v2
cublasCsyr2 = libcublas.cublasCsyr2_v2
cublasZsyr2 = libcublas.cublasZsyr2_v2
for funct in [cublasSsyr2, cublasDsyr2, cublasCsyr2, cublasZsyr2]:
    funct.restype = cublasStatus_t
    funct.argtypes = [cublasHandle_t,
                      c_cublasFillMode_t,  # uplo
                      c_int,  # n
                      scalar_pointer,  # *alpha
                      memory_pointer, c_int,  # *x, incx
                      memory_pointer, c_int,  # *y, incy
                      memory_pointer, c_int  # *A, lda
                      ]

# cublasStatus_t cublasStbmv(cublasHandle_t handle, cublasFillMode_t uplo,
#                            cublasOperation_t trans, cublasDiagType_t diag,
#                            int n, int k, const float           *A, int lda,
#                            float           *x, int incx)
# cublasStatus_t cublasDtbmv(cublasHandle_t handle, cublasFillMode_t uplo,
#                            cublasOperation_t trans, cublasDiagType_t diag,
#                            int n, int k, const double          *A, int lda,
#                            double          *x, int incx)
# cublasStatus_t cublasCtbmv(cublasHandle_t handle, cublasFillMode_t uplo,
#                            cublasOperation_t trans, cublasDiagType_t diag,
#                            int n, int k, const cuComplex       *A, int lda,
#                            cuComplex       *x, int incx)
# cublasStatus_t cublasZtbmv(cublasHandle_t handle, cublasFillMode_t uplo,
#                            cublasOperation_t trans, cublasDiagType_t diag,
#                            int n, int k, const cuDoubleComplex *A, int lda,
#                            cuDoubleComplex *x, int incx)
cublasStbmv = libcublas.cublasStbmv_v2
cublasDtbmv = libcublas.cublasDtbmv_v2
cublasCtbmv = libcublas.cublasCtbmv_v2
cublasZtbmv = libcublas.cublasZtbmv_v2
for funct in [cublasStbmv, cublasDtbmv, cublasCtbmv, cublasZtbmv]:
    funct.restype = cublasStatus_t
    funct.argtypes = [cublasHandle_t,
                      c_cublasFillMode_t,  # uplo
                      c_cublasOperation_t,  # trans
                      c_cublasDiagType_t,  # diag
                      c_int,  c_int,  # n, k
                      memory_pointer, c_int,  # *A, lda
                      memory_pointer, c_int  # *x, incx
                      ]

# cublasStatus_t cublasStbsv(cublasHandle_t handle, cublasFillMode_t uplo,
#                            cublasOperation_t trans, cublasDiagType_t diag,
#                            int n, int k, const float           *A, int lda,
#                            float           *x, int incx)
# cublasStatus_t cublasDtbsv(cublasHandle_t handle, cublasFillMode_t uplo,
#                            cublasOperation_t trans, cublasDiagType_t diag,
#                            int n, int k, const double          *A, int lda,
#                            double          *x, int incx)
# cublasStatus_t cublasCtbsv(cublasHandle_t handle, cublasFillMode_t uplo,
#                            cublasOperation_t trans, cublasDiagType_t diag,
#                            int n, int k, const cuComplex       *A, int lda,
#                            cuComplex       *x, int incx)
# cublasStatus_t cublasZtbsv(cublasHandle_t handle, cublasFillMode_t uplo,
#                            cublasOperation_t trans, cublasDiagType_t diag,
#                            int n, int k, const cuDoubleComplex *A, int lda,
#                            cuDoubleComplex *x, int incx)
cublasStbsv = libcublas.cublasStbsv_v2
cublasDtbsv = libcublas.cublasDtbsv_v2
cublasCtbsv = libcublas.cublasCtbsv_v2
cublasZtbsv = libcublas.cublasZtbsv_v2
for funct in [cublasStbsv, cublasDtbsv, cublasCtbsv, cublasZtbsv]:
    funct.restype = cublasStatus_t
    funct.argtypes = [cublasHandle_t,
                      c_cublasFillMode_t,  # uplo
                      c_cublasOperation_t,  # trans
                      c_cublasDiagType_t,  # diag
                      c_int,  c_int,  # n, k
                      memory_pointer, c_int,  # *A, lda
                      memory_pointer, c_int  # *x, incx
                      ]

# cublasStatus_t cublasStpmv(cublasHandle_t handle, cublasFillMode_t uplo,
#                            cublasOperation_t trans, cublasDiagType_t diag,
#                            int n, const float           *AP,
#                            float           *x, int incx)
# cublasStatus_t cublasDtpmv(cublasHandle_t handle, cublasFillMode_t uplo,
#                            cublasOperation_t trans, cublasDiagType_t diag,
#                            int n, const double          *AP,
#                            double          *x, int incx)
# cublasStatus_t cublasCtpmv(cublasHandle_t handle, cublasFillMode_t uplo,
#                            cublasOperation_t trans, cublasDiagType_t diag,
#                            int n, const cuComplex       *AP,
#                            cuComplex       *x, int incx)
# cublasStatus_t cublasZtpmv(cublasHandle_t handle, cublasFillMode_t uplo,
#                            cublasOperation_t trans, cublasDiagType_t diag,
#                            int n, const cuDoubleComplex *AP,
#                            cuDoubleComplex *x, int incx)
cublasStpmv = libcublas.cublasStpmv_v2
cublasDtpmv = libcublas.cublasDtpmv_v2
cublasCtpmv = libcublas.cublasCtpmv_v2
cublasZtpmv = libcublas.cublasZtpmv_v2
for funct in [cublasStpmv, cublasDtpmv, cublasCtpmv, cublasZtpmv]:
    funct.restype = cublasStatus_t
    funct.argtypes = [cublasHandle_t,
                      c_cublasFillMode_t,  # uplo
                      c_cublasOperation_t,  # trans
                      c_cublasDiagType_t,  # diag
                      c_int,  # n
                      memory_pointer,  # *AP
                      memory_pointer, c_int  # *x, incx
                      ]

# cublasStatus_t cublasStpsv(cublasHandle_t handle, cublasFillMode_t uplo,
#                            cublasOperation_t trans, cublasDiagType_t diag,
#                            int n, const float           *AP,
#                            #float           *x, int incx)
# cublasStatus_t cublasDtpsv(cublasHandle_t handle, cublasFillMode_t uplo,
#                            cublasOperation_t trans, cublasDiagType_t diag,
#                            int n, const double          *AP,
#                            double          *x, int incx)
# cublasStatus_t cublasCtpsv(cublasHandle_t handle, cublasFillMode_t uplo,
#                            cublasOperation_t trans, cublasDiagType_t diag,
#                            int n, const cuComplex       *AP,
#                            cuComplex       *x, int incx)
# cublasStatus_t cublasZtpsv(cublasHandle_t handle, cublasFillMode_t uplo,
#                            cublasOperation_t trans, cublasDiagType_t diag,
#                            int n, const cuDoubleComplex *AP,
#                            cuDoubleComplex *x, int incx)
cublasStpsv = libcublas.cublasStpsv_v2
cublasDtpsv = libcublas.cublasDtpsv_v2
cublasCtpsv = libcublas.cublasCtpsv_v2
cublasZtpsv = libcublas.cublasZtpsv_v2
for funct in [cublasStpsv, cublasDtpsv, cublasCtpsv, cublasZtpsv]:
    funct.restype = cublasStatus_t
    funct.argtypes = [cublasHandle_t,
                      c_cublasFillMode_t,  # uplo
                      c_cublasOperation_t,  # trans
                      c_cublasDiagType_t,  # diag
                      c_int,  # n
                      memory_pointer,  # *AP
                      memory_pointer, c_int  # *x, incx
                      ]

# cublasStatus_t cublasStrmv(cublasHandle_t handle, cublasFillMode_t uplo,
#                            cublasOperation_t trans, cublasDiagType_t diag,
#                            int n, const float           *A, int lda,
#                            float           *x, int incx)
# cublasStatus_t cublasDtrmv(cublasHandle_t handle, cublasFillMode_t uplo,
#                            cublasOperation_t trans, cublasDiagType_t diag,
#                            int n, const double          *A, int lda,
#                            double          *x, int incx)
# cublasStatus_t cublasCtrmv(cublasHandle_t handle, cublasFillMode_t uplo,
#                            cublasOperation_t trans, cublasDiagType_t diag,
#                            int n, const cuComplex       *A, int lda,
#                            cuComplex       *x, int incx)
# cublasStatus_t cublasZtrmv(cublasHandle_t handle, cublasFillMode_t uplo,
#                            cublasOperation_t trans, cublasDiagType_t diag,
#                            int n, const cuDoubleComplex *A, int lda,
#                            cuDoubleComplex *x, int incx)
cublasStrmv = libcublas.cublasStrmv_v2
cublasDtrmv = libcublas.cublasDtrmv_v2
cublasCtrmv = libcublas.cublasCtrmv_v2
cublasZtrmv = libcublas.cublasZtrmv_v2
for funct in [cublasStrmv, cublasDtrmv, cublasCtrmv, cublasZtrmv]:
    funct.restype = cublasStatus_t
    funct.argtypes = [cublasHandle_t,
                      c_cublasFillMode_t,  # uplo
                      c_cublasOperation_t,  # trans
                      c_cublasDiagType_t,  # diag
                      c_int,  # n
                      memory_pointer, c_int,  # *A, lda
                      memory_pointer, c_int  # *x, incx
                      ]

# cublasStatus_t cublasStrsv(cublasHandle_t handle, cublasFillMode_t uplo,
#                            cublasOperation_t trans, cublasDiagType_t diag,
#                            int n, const float           *A, int lda,
#                            float           *x, int incx)
# cublasStatus_t cublasDtrsv(cublasHandle_t handle, cublasFillMode_t uplo,
#                            cublasOperation_t trans, cublasDiagType_t diag,
#                            int n, const double          *A, int lda,
#                            double          *x, int incx)
# cublasStatus_t cublasCtrsv(cublasHandle_t handle, cublasFillMode_t uplo,
#                            cublasOperation_t trans, cublasDiagType_t diag,
#                            int n, const cuComplex       *A, int lda,
#                            cuComplex       *x, int incx)
# cublasStatus_t cublasZtrsv(cublasHandle_t handle, cublasFillMode_t uplo,
#                            cublasOperation_t trans, cublasDiagType_t diag,
#                            int n, const cuDoubleComplex *A, int lda,
#                            cuDoubleComplex *x, int incx)
cublasStrsv = libcublas.cublasStrsv_v2
cublasDtrsv = libcublas.cublasDtrsv_v2
cublasCtrsv = libcublas.cublasCtrsv_v2
cublasZtrsv = libcublas.cublasZtrsv_v2
for funct in [cublasStrsv, cublasDtrsv, cublasCtrsv, cublasZtrsv]:
    funct.restype = cublasStatus_t
    funct.argtypes = [cublasHandle_t,
                      c_cublasFillMode_t,  # uplo
                      c_cublasOperation_t,  # trans
                      c_cublasDiagType_t,  # diag
                      c_int,  # n
                      memory_pointer, c_int,  # *A, lda
                      memory_pointer, c_int  # *x, incx
                      ]

# cublasStatus_t cublasChemv(cublasHandle_t handle, cublasFillMode_t uplo,
#                            int n, const cuComplex       *alpha,
#                            const cuComplex       *A, int lda,
#                            const cuComplex       *x, int incx,
#                            const cuComplex       *beta,
#                            cuComplex       *y, int incy)
# cublasStatus_t cublasZhemv(cublasHandle_t handle, cublasFillMode_t uplo,
#                            int n, const cuDoubleComplex *alpha,
#                            const cuDoubleComplex *A, int lda,
#                            const cuDoubleComplex *x, int incx,
#                            const cuDoubleComplex *beta,
#                            cuDoubleComplex *y, int incy)
cublasChemv = libcublas.cublasChemv_v2
cublasZhemv = libcublas.cublasZhemv_v2
for funct in [cublasChemv, cublasZhemv]:
    funct.restype = cublasStatus_t
    funct.argtypes = [cublasHandle_t,
                      c_cublasFillMode_t,  # uplo
                      c_int,  # n
                      scalar_pointer,  # *alpha
                      memory_pointer, c_int,  # *A, lda
                      memory_pointer, c_int,  # *x, incx
                      scalar_pointer,  # *beta
                      memory_pointer, c_int  # *y, incy
                      ]

# cublasStatus_t cublasChbmv(cublasHandle_t handle, cublasFillMode_t uplo,
#                           int n, int k, const cuComplex       *alpha,
#                           const cuComplex       *A, int lda,
#                           const cuComplex       *x, int incx,
#                           const cuComplex       *beta,
#                           cuComplex       *y, int incy)
# cublasStatus_t cublasZhbmv(cublasHandle_t handle, cublasFillMode_t uplo,
#                           int n, int k, const cuDoubleComplex *alpha,
#                           const cuDoubleComplex *A, int lda,
#                           const cuDoubleComplex *x, int incx,
#                           const cuDoubleComplex *beta,
#                           cuDoubleComplex *y, int incy)
cublasChbmv = libcublas.cublasChbmv_v2
cublasZhbmv = libcublas.cublasZhbmv_v2
for funct in [cublasChbmv, cublasZhbmv]:
    funct.restype = cublasStatus_t
    funct.argtypes = [cublasHandle_t,
                      c_cublasFillMode_t,  # uplo
                      c_int, c_int,  # n, k
                      scalar_pointer,  # *alpha
                      memory_pointer, c_int,  # *A, lda
                      memory_pointer, c_int,  # *x, incx
                      scalar_pointer,  # *beta
                      memory_pointer, c_int  # *y, incy
                      ]

# cublasStatus_t cublasChpmv(cublasHandle_t handle, cublasFillMode_t uplo,
#                            int n, const cuComplex       *alpha,
#                            const cuComplex       *AP,
#                            const cuComplex       *x, int incx,
#                            const cuComplex       *beta,
#                            cuComplex       *y, int incy)
# cublasStatus_t cublasZhpmv(cublasHandle_t handle, cublasFillMode_t uplo,
#                            int n, const cuDoubleComplex *alpha,
#                            const cuDoubleComplex *AP,
#                            const cuDoubleComplex *x, int incx,
#                            const cuDoubleComplex *beta,
#                            cuDoubleComplex *y, int incy)
cublasChpmv = libcublas.cublasChpmv_v2
cublasZhpmv = libcublas.cublasZhpmv_v2
for funct in [cublasChpmv, cublasZhpmv]:
    funct.restype = cublasStatus_t
    funct.argtypes = [cublasHandle_t,
                      c_cublasFillMode_t,  # uplo
                      c_int,  # n
                      scalar_pointer,  # *alpha
                      memory_pointer,  # *AP
                      memory_pointer, c_int,  # *x, incx
                      scalar_pointer,  # *beta
                      memory_pointer, c_int  # *y, incy
                      ]

# cublasStatus_t cublasCher(cublasHandle_t handle, cublasFillMode_t uplo,
#                           int n, const float  *alpha,
#                           const cuComplex       *x, int incx,
#                           cuComplex       *A, int lda)
# cublasStatus_t cublasZher(cublasHandle_t handle, cublasFillMode_t uplo,
#                           int n, const double *alpha,
#                           const cuDoubleComplex *x, int incx,
#                           cuDoubleComplex *A, int lda)
cublasCher = libcublas.cublasCher_v2
cublasZher = libcublas.cublasZher_v2
for funct in [cublasCher, cublasZher]:
    funct.restype = cublasStatus_t
    funct.argtypes = [cublasHandle_t,
                      c_cublasFillMode_t,  # uplo
                      c_int,  # n
                      scalar_pointer,  # *alpha
                      memory_pointer, c_int,  # *x, incx
                      memory_pointer, c_int  # *A, lda
                      ]

# cublasStatus_t cublasCher2(cublasHandle_t handle, cublasFillMode_t uplo,
#                            int n, const cuComplex       *alpha,
#                            const cuComplex       *x, int incx,
#                            const cuComplex       *y, int incy,
#                            cuComplex       *A, int lda)
# cublasStatus_t cublasZher2(cublasHandle_t handle, cublasFillMode_t uplo,
#                            int n, const cuDoubleComplex *alpha,
#                            const cuDoubleComplex *x, int incx,
#                            const cuDoubleComplex *y, int incy,
#                            cuDoubleComplex *A, int lda)
cublasCher2 = libcublas.cublasCher2_v2
cublasZher2 = libcublas.cublasZher2_v2
for funct in [cublasCher2, cublasZher2]:
    funct.restype = cublasStatus_t
    funct.argtypes = [cublasHandle_t,
                      c_cublasFillMode_t,  # uplo
                      c_int,  # n
                      scalar_pointer,  # *alpha
                      memory_pointer, c_int,  # *x, incx
                      memory_pointer, c_int,  # *y, incy
                      memory_pointer, c_int  # *A, lda
                      ]

# cublasStatus_t cublasChpr(cublasHandle_t handle, cublasFillMode_t uplo,
#                           int n, const float *alpha,
#                           const cuComplex       *x, int incx,
#                           cuComplex       *AP)
# cublasStatus_t cublasZhpr(cublasHandle_t handle, cublasFillMode_t uplo,
#                           int n, const double *alpha,
#                           const cuDoubleComplex *x, int incx,
#                           cuDoubleComplex *AP)
cublasChpr = libcublas.cublasChpr_v2
cublasZhpr = libcublas.cublasZhpr_v2
for funct in [cublasChpr, cublasZhpr]:
    funct.restype = cublasStatus_t
    funct.argtypes = [cublasHandle_t,
                      c_cublasFillMode_t,  # uplo
                      c_int,  # n
                      scalar_pointer,  # alpha
                      memory_pointer, c_int,  # *x, incx
                      memory_pointer  # *AP
                      ]

# cublasStatus_t cublasChpr2(cublasHandle_t handle, cublasFillMode_t uplo,
#                            int n, const cuComplex       *alpha,
#                            const cuComplex       *x, int incx,
#                            const cuComplex       *y, int incy,
#                            cuComplex       *AP)
# cublasStatus_t cublasZhpr2(cublasHandle_t handle, cublasFillMode_t uplo,
#                            int n, const cuDoubleComplex *alpha,
#                            const cuDoubleComplex *x, int incx,
#                            const cuDoubleComplex *y, int incy,
#                            cuDoubleComplex *AP)
cublasChpr2 = libcublas.cublasChpr2_v2
cublasZhpr2 = libcublas.cublasZhpr2_v2
for funct in [cublasChpr2, cublasZhpr2]:
    funct.restype = cublasStatus_t
    funct.argtypes = [cublasHandle_t,
                      c_cublasFillMode_t,  # uplo
                      c_int,  # n
                      scalar_pointer,  # *alpha
                      memory_pointer, c_int,  # *x, incx
                      memory_pointer, c_int,  # *y, incy
                      memory_pointer  # *AP
                      ]

## cuBLAS Level-3 Functions ##

# cublasStatus_t cublasSgemm(cublasHandle_t handle,
#                            cublasOperation_t transa, cublasOperation_t transb,
#                            int m, int n, int k,
#                            const float           *alpha,
#                            const float           *A, int lda,
#                            const float           *B, int ldb,
#                            const float           *beta,
#                            float           *C, int ldc)
# cublasStatus_t cublasDgemm(cublasHandle_t handle,
#                            cublasOperation_t transa, cublasOperation_t transb,
#                            int m, int n, int k,
#                            const double          *alpha,
#                            const double          *A, int lda,
#                            const double          *B, int ldb,
#                            const double          *beta,
#                            double          *C, int ldc)
# cublasStatus_t cublasCgemm(cublasHandle_t handle,
#                            cublasOperation_t transa, cublasOperation_t transb,
#                            int m, int n, int k,
#                            const cuComplex       *alpha,
#                            const cuComplex       *A, int lda,
#                            const cuComplex       *B, int ldb,
#                            const cuComplex       *beta,
#                            cuComplex       *C, int ldc)
# cublasStatus_t cublasZgemm(cublasHandle_t handle,
#                            cublasOperation_t transa, cublasOperation_t transb,
#                            int m, int n, int k,
#                            const cuDoubleComplex *alpha,
#                            const cuDoubleComplex *A, int lda,
#                            const cuDoubleComplex *B, int ldb,
#                            const cuDoubleComplex *beta,
#                            cuDoubleComplex *C, int ldc)
cublasSgemm = libcublas.cublasSgemm_v2
cublasDgemm = libcublas.cublasDgemm_v2
cublasCgemm = libcublas.cublasCgemm_v2
cublasZgemm = libcublas.cublasZgemm_v2
for funct in [cublasSgemm, cublasDgemm, cublasCgemm, cublasZgemm]:
    funct.restype = cublasStatus_t
    funct.argtypes = [cublasHandle_t,
                      c_cublasOperation_t,  # transa
                      c_cublasOperation_t,  # transb
                      c_int, c_int, c_int,  # m, n, k
                      scalar_pointer,  # *alpha
                      memory_pointer, c_int,  # *A, lda
                      memory_pointer, c_int,  # *B, ldb
                      scalar_pointer,  # *beta
                      memory_pointer, c_int  # *C, ldc
                      ]

# cublasStatus_t cublasSgemmBatched(cublasHandle_t handle,
#                                   cublasOperation_t transa, cublasOperation_t transb,
#                                   int m, int n, int k,
#                                   const float           *alpha,
#                                   const float           *Aarray[], int lda,
#                                   const float           *Barray[], int ldb,
#                                   const float           *beta,
#                                   float           *Carray[], int ldc, int batchCount)
# cublasStatus_t cublasDgemmBatched(cublasHandle_t handle,
#                                   cublasOperation_t transa, cublasOperation_t transb,
#                                   int m, int n, int k,
#                                   const double          *alpha,
#                                   const double          *Aarray[], int lda,
#                                   const double          *Barray[], int ldb,
#                                   const double          *beta,
#                                   double          *Carray[], int ldc, int batchCount)
# cublasStatus_t cublasCgemmBatched(cublasHandle_t handle,
#                                   cublasOperation_t transa, cublasOperation_t transb,
#                                   int m, int n, int k,
#                                   const cuComplex       *alpha,
#                                   const cuComplex       *Aarray[], int lda,
#                                   const cuComplex       *Barray[], int ldb,
#                                   const cuComplex       *beta,
#                                   cuComplex       *Carray[], int ldc, int batchCount)
# cublasStatus_t cublasZgemmBatched(cublasHandle_t handle,
#                                   cublasOperation_t transa, cublasOperation_t transb,
#                                   int m, int n, int k,
#                                   const cuDoubleComplex *alpha,
#                                   const cuDoubleComplex *Aarray[], int lda,
#                                   const cuDoubleComplex *Barray[], int ldb,
#                                   const cuDoubleComplex *beta,
#                                   cuDoubleComplex *Carray[], int ldc, int batchCount)
cublasSgemmBatched = libcublas.cublasSgemmBatched
cublasDgemmBatched = libcublas.cublasDgemmBatched
cublasCgemmBatched = libcublas.cublasCgemmBatched
cublasZgemmBatched = libcublas.cublasZgemmBatched
for funct in [cublasSgemmBatched, cublasDgemmBatched,
              cublasCgemmBatched, cublasZgemmBatched]:
    funct.restype = cublasStatus_t
    funct.argtypes = [cublasHandle_t,
                      c_cublasOperation_t,  # transa
                      c_cublasOperation_t,  # transb
                      c_int, c_int, c_int,  # m, n, k
                      scalar_pointer,  # *alpha
                      array_pointer, c_int,  # *A[], lda
                      array_pointer, c_int,  # *B[], ldb
                      scalar_pointer,  # *beta
                      array_pointer, c_int,  # *C[], ldc
                      c_int  # batchCount
                      ]

# cublasStatus_t cublasSsymm(cublasHandle_t handle,
#                            cublasSideMode_t side, cublasFillMode_t uplo,
#                            int m, int n,
#                            const float           *alpha,
#                            const float           *A, int lda,
#                            const float           *B, int ldb,
#                            const float           *beta,
#                            float           *C, int ldc)
# cublasStatus_t cublasDsymm(cublasHandle_t handle,
#                            cublasSideMode_t side, cublasFillMode_t uplo,
#                            int m, int n,
#                            const double          *alpha,
#                            const double          *A, int lda,
#                            const double          *B, int ldb,
#                            const double          *beta,
#                            double          *C, int ldc)
# cublasStatus_t cublasCsymm(cublasHandle_t handle,
#                            cublasSideMode_t side, cublasFillMode_t uplo,
#                            int m, int n,
#                            const cuComplex       *alpha,
#                            const cuComplex       *A, int lda,
#                            const cuComplex       *B, int ldb,
#                            const cuComplex       *beta,
#                            cuComplex       *C, int ldc)
# cublasStatus_t cublasZsymm(cublasHandle_t handle,
#                            cublasSideMode_t side, cublasFillMode_t uplo,
#                            int m, int n,
#                            const cuDoubleComplex *alpha,
#                            const cuDoubleComplex *A, int lda,
#                            const cuDoubleComplex *B, int ldb,
#                            const cuDoubleComplex *beta,
#                            cuDoubleComplex *C, int ldc)
cublasSsymm = libcublas.cublasSsymm_v2
cublasDsymm = libcublas.cublasDsymm_v2
cublasCsymm = libcublas.cublasCsymm_v2
cublasZsymm = libcublas.cublasZsymm_v2
for funct in [cublasSsymm, cublasDsymm, cublasCsymm, cublasZsymm]:
    funct.restype = cublasStatus_t
    funct.argtypes = [cublasHandle_t,
                      c_cublasSideMode_t,  # side
                      c_cublasFillMode_t,  # uplo
                      c_int, c_int,  # m, n
                      scalar_pointer,  # *alpha
                      memory_pointer, c_int,  # *A, lda
                      memory_pointer, c_int,  # *B, ldb
                      scalar_pointer,  # *beta
                      memory_pointer, c_int  # *C, ldc
                      ]

# cublasStatus_t cublasSsyrk(cublasHandle_t handle,
#                            cublasFillMode_t uplo, cublasOperation_t trans,
#                            int n, int k,
#                            const float           *alpha,
#                            const float           *A, int lda,
#                            const float           *beta,
#                            float           *C, int ldc)
# cublasStatus_t cublasDsyrk(cublasHandle_t handle,
#                            cublasFillMode_t uplo, cublasOperation_t trans,
#                            int n, int k,
#                            const double          *alpha,
#                            const double          *A, int lda,
#                            const double          *beta,
#                            double          *C, int ldc)
# cublasStatus_t cublasCsyrk(cublasHandle_t handle,
#                            cublasFillMode_t uplo, cublasOperation_t trans,
#                            int n, int k,
#                            const cuComplex       *alpha,
#                            const cuComplex       *A, int lda,
#                            const cuComplex       *beta,
#                            cuComplex       *C, int ldc)
# cublasStatus_t cublasZsyrk(cublasHandle_t handle,
#                            cublasFillMode_t uplo, cublasOperation_t trans,
#                            int n, int k,
#                            const cuDoubleComplex *alpha,
#                            const cuDoubleComplex *A, int lda,
#                            const cuDoubleComplex *beta,
#                            cuDoubleComplex *C, int ldc)
cublasSsyrk = libcublas.cublasSsyrk_v2
cublasDsyrk = libcublas.cublasDsyrk_v2
cublasCsyrk = libcublas.cublasCsyrk_v2
cublasZsyrk = libcublas.cublasZsyrk_v2
for funct in [cublasSsyrk, cublasDsyrk, cublasCsyrk, cublasZsyrk]:
    funct.restype = cublasStatus_t
    funct.argtypes = [cublasHandle_t,
                      c_cublasFillMode_t,  # uplo
                      c_cublasOperation_t,  # trans
                      c_int, c_int,  # n, k
                      scalar_pointer,  # *alpha
                      memory_pointer, c_int,  # *A, lda
                      scalar_pointer,  # *beta
                      memory_pointer, c_int  # *C, ldc
                      ]

# cublasStatus_t cublasSsyr2k(cublasHandle_t handle,
#                             cublasFillMode_t uplo, cublasOperation_t trans,
#                             int n, int k,
#                             const float           *alpha,
#                             const float           *A, int lda,
#                             const float           *B, int ldb,
#                             const float           *beta,
#                             float           *C, int ldc)
# cublasStatus_t cublasDsyr2k(cublasHandle_t handle,
#                             cublasFillMode_t uplo, cublasOperation_t trans,
#                             int n, int k,
#                             const double          *alpha,
#                             const double          *A, int lda,
#                             const double          *B, int ldb,
#                             const double          *beta,
#                             double          *C, int ldc)
# cublasStatus_t cublasCsyr2k(cublasHandle_t handle,
#                             cublasFillMode_t uplo, cublasOperation_t trans,
#                             int n, int k,
#                             const cuComplex       *alpha,
#                             const cuComplex       *A, int lda,
#                             const cuComplex       *B, int ldb,
#                             const cuComplex       *beta,
#                             cuComplex       *C, int ldc)
# cublasStatus_t cublasZsyr2k(cublasHandle_t handle,
#                             cublasFillMode_t uplo, cublasOperation_t trans,
#                             int n, int k,
#                             const cuDoubleComplex *alpha,
#                             const cuDoubleComplex *A, int lda,
#                             const cuDoubleComplex *B, int ldb,
#                             const cuDoubleComplex *beta,
#                             cuDoubleComplex *C, int ldc)
cublasSsyr2k = libcublas.cublasSsyr2k_v2
cublasDsyr2k = libcublas.cublasDsyr2k_v2
cublasCsyr2k = libcublas.cublasCsyr2k_v2
cublasZsyr2k = libcublas.cublasZsyr2k_v2
for funct in [cublasSsyr2k, cublasDsyr2k, cublasCsyr2k, cublasZsyr2k]:
    funct.restype = cublasStatus_t
    funct.argtypes = [cublasHandle_t,
                      c_cublasFillMode_t,  # uplo
                      c_cublasOperation_t,  # trans
                      c_int, c_int,  # n, k
                      scalar_pointer,  # *alpha
                      memory_pointer, c_int,  # *A, lda
                      memory_pointer, c_int,  # *B, ldb
                      scalar_pointer,  # *beta
                      memory_pointer, c_int  # *C, ldc
                      ]

# cublasStatus_t cublasSsyrkx(cublasHandle_t handle,
#                             cublasFillMode_t uplo, cublasOperation_t trans,
#                             int n, int k,
#                             const float           *alpha,
#                             const float           *A, int lda,
#                             const float           *B, int ldb,
#                             const float           *beta,
#                             float           *C, int ldc)
# cublasStatus_t cublasDsyrkx(cublasHandle_t handle,
#                             cublasFillMode_t uplo, cublasOperation_t trans,
#                             int n, int k,
#                             const double          *alpha,
#                             const double          *A, int lda,
#                             const double          *B, int ldb,
#                             const double          *beta,
#                             double          *C, int ldc)
# cublasStatus_t cublasCsyrkx(cublasHandle_t handle,
#                             cublasFillMode_t uplo, cublasOperation_t trans,
#                             int n, int k,
#                             const cuComplex       *alpha,
#                             const cuComplex       *A, int lda,
#                             const cuComplex       *B, int ldb,
#                             const cuComplex       *beta,
#                             cuComplex       *C, int ldc)
# cublasStatus_t cublasZsyrkx(cublasHandle_t handle,
#                             cublasFillMode_t uplo, cublasOperation_t trans,
#                             int n, int k,
#                             const cuDoubleComplex *alpha,
#                             const cuDoubleComplex *A, int lda,
#                             const cuDoubleComplex *B, int ldb,
#                             const cuDoubleComplex *beta,
#                             cuDoubleComplex *C, int ldc)
cublasSsyrkx = libcublas.cublasSsyrkx
cublasDsyrkx = libcublas.cublasDsyrkx
cublasCsyrkx = libcublas.cublasCsyrkx
cublasZsyrkx = libcublas.cublasZsyrkx
for funct in [cublasSsyrkx, cublasDsyrkx, cublasCsyrkx, cublasZsyrkx]:
    funct.restype = cublasStatus_t
    funct.argtypes = [cublasHandle_t,
                      c_cublasFillMode_t,  # uplo
                      c_cublasOperation_t,  # trans
                      c_int, c_int,  # n, k
                      scalar_pointer,  # *alpha
                      memory_pointer, c_int,  # *A, lda
                      memory_pointer, c_int,  # *B, ldb
                      scalar_pointer,  # *beta
                      memory_pointer, c_int  # *C, ldc
                      ]

# cublasStatus_t cublasStrmm(cublasHandle_t handle,
#                            cublasSideMode_t side, cublasFillMode_t uplo,
#                            cublasOperation_t trans, cublasDiagType_t diag,
#                            int m, int n,
#                            const float           *alpha,
#                            const float           *A, int lda,
#                            const float           *B, int ldb,
#                            float                 *C, int ldc)
# cublasStatus_t cublasDtrmm(cublasHandle_t handle,
#                            cublasSideMode_t side, cublasFillMode_t uplo,
#                            cublasOperation_t trans, cublasDiagType_t diag,
#                            int m, int n,
#                            const double          *alpha,
#                            const double          *A, int lda,
#                            const double          *B, int ldb,
#                            double                *C, int ldc)
# cublasStatus_t cublasCtrmm(cublasHandle_t handle,
#                            cublasSideMode_t side, cublasFillMode_t uplo,
#                            cublasOperation_t trans, cublasDiagType_t diag,
#                            int m, int n,
#                            const cuComplex       *alpha,
#                            const cuComplex       *A, int lda,
#                            const cuComplex       *B, int ldb,
#                            cuComplex             *C, int ldc)
# cublasStatus_t cublasZtrmm(cublasHandle_t handle,
#                            cublasSideMode_t side, cublasFillMode_t uplo,
#                            cublasOperation_t trans, cublasDiagType_t diag,
#                            int m, int n,
#                            const cuDoubleComplex *alpha,
#                            const cuDoubleComplex *A, int lda,
#                            const cuDoubleComplex *B, int ldb,
#                            cuDoubleComplex       *C, int ldc)
cublasStrmm = libcublas.cublasStrmm_v2
cublasDtrmm = libcublas.cublasDtrmm_v2
cublasCtrmm = libcublas.cublasCtrmm_v2
cublasZtrmm = libcublas.cublasZtrmm_v2
for funct in [cublasStrmm, cublasDtrmm, cublasCtrmm, cublasZtrmm]:
    funct.restype = cublasStatus_t
    funct.argtypes = [cublasHandle_t,
                      c_cublasSideMode_t,  # side
                      c_cublasFillMode_t,  # uplo
                      c_cublasOperation_t,  # trans
                      c_cublasDiagType_t,  # diag
                      c_int, c_int,  # m, n
                      scalar_pointer,  # *alpha
                      memory_pointer, c_int,  # *A, lda
                      memory_pointer, c_int,  # *B, ldb
                      memory_pointer, c_int  # *C, ldc
                      ]

# cublasStatus_t cublasStrsm(cublasHandle_t handle,
#                            cublasSideMode_t side, cublasFillMode_t uplo,
#                            cublasOperation_t trans, cublasDiagType_t diag,
#                            int m, int n,
#                            const float           *alpha,
#                            const float           *A, int lda,
#                            float           *B, int ldb)
# cublasStatus_t cublasDtrsm(cublasHandle_t handle,
#                            cublasSideMode_t side, cublasFillMode_t uplo,
#                            cublasOperation_t trans, cublasDiagType_t diag,
#                            int m, int n,
#                            const double          *alpha,
#                            const double          *A, int lda,
#                            double          *B, int ldb)
# cublasStatus_t cublasCtrsm(cublasHandle_t handle,
#                            cublasSideMode_t side, cublasFillMode_t uplo,
#                            cublasOperation_t trans, cublasDiagType_t diag,
#                            int m, int n,
#                            const cuComplex       *alpha,
#                            const cuComplex       *A, int lda,
#                            cuComplex       *B, int ldb)
# cublasStatus_t cublasZtrsm(cublasHandle_t handle,
#                            cublasSideMode_t side, cublasFillMode_t uplo,
#                            cublasOperation_t trans, cublasDiagType_t diag,
#                            int m, int n,
#                            const cuDoubleComplex *alpha,
#                            const cuDoubleComplex *A, int lda,
#                            cuDoubleComplex *B, int ldb)
cublasStrsm = libcublas.cublasStrsm_v2
cublasDtrsm = libcublas.cublasDtrsm_v2
cublasCtrsm = libcublas.cublasCtrsm_v2
cublasZtrsm = libcublas.cublasZtrsm_v2
for funct in [cublasStrsm, cublasDtrsm, cublasCtrsm, cublasZtrsm]:
    funct.restype = cublasStatus_t
    funct.argtypes = [cublasHandle_t,
                      c_cublasSideMode_t,  # side
                      c_cublasFillMode_t,  # uplo
                      c_cublasOperation_t,  # trans
                      c_cublasDiagType_t,  # diag
                      c_int, c_int,  # m, n
                      scalar_pointer,  # *alpha
                      memory_pointer, c_int,  # *A, lda
                      memory_pointer, c_int  # *B, ldb
                      ]

# cublasStatus_t cublasStrsmBatched( cublasHandle_t    handle,
#                                    cublasSideMode_t  side,
#                                    cublasFillMode_t  uplo,
#                                    cublasOperation_t trans,
#                                    cublasDiagType_t  diag,
#                                    int m,
#                                    int n,
#                                    const float *alpha,
#                                    float *A[],
#                                    int lda,
#                                    float *B[],
#                                    int ldb,
#                                    int batchCount);
# cublasStatus_t cublasDtrsmBatched( cublasHandle_t    handle,
#                                    cublasSideMode_t  side,
#                                    cublasFillMode_t  uplo,
#                                    cublasOperation_t trans,
#                                    cublasDiagType_t  diag,
#                                    int m,
#                                    int n,
#                                    const double *alpha,
#                                    double *A[],
#                                    int lda,
#                                    double *B[],
#                                    int ldb,
#                                    int batchCount);
# cublasStatus_t cublasCtrsmBatched( cublasHandle_t    handle,
#                                    cublasSideMode_t  side,
#                                    cublasFillMode_t  uplo,
#                                    cublasOperation_t trans,
#                                    cublasDiagType_t  diag,
#                                    int m,
#                                    int n,
#                                    const cuComplex *alpha,
#                                    cuComplex *A[],
#                                    int lda,
#                                    cuComplex *B[],
#                                    int ldb,
#                                    int batchCount);
# cublasStatus_t cublasZtrsmBatched( cublasHandle_t    handle,
#                                    cublasSideMode_t  side,
#                                    cublasFillMode_t  uplo,
#                                    cublasOperation_t trans,
#                                    cublasDiagType_t  diag,
#                                    int m,
#                                    int n,
#                                    const cuDoubleComplex *alpha,
#                                    cuDoubleComplex *A[],
#                                    int lda,
#                                    cuDoubleComplex *B[],
#                                    int ldb,
#                                    int batchCount);
cublasStrsmBatched = libcublas.cublasStrsmBatched
cublasDtrsmBatched = libcublas.cublasDtrsmBatched
cublasCtrsmBatched = libcublas.cublasCtrsmBatched
cublasZtrsmBatched = libcublas.cublasZtrsmBatched
for funct in [cublasStrsmBatched, cublasDtrsmBatched,
              cublasCtrsmBatched, cublasZtrsmBatched]:
    funct.restype = cublasStatus_t
    funct.argtypes = [cublasHandle_t,
                      c_cublasSideMode_t,  # side
                      c_cublasFillMode_t,  # uplo
                      c_cublasOperation_t,  # trans
                      c_cublasDiagType_t,  # diag
                      c_int, c_int,  # m, n
                      scalar_pointer,  # *alpha
                      array_pointer, c_int,  # *A[], lda
                      array_pointer, c_int,  # *B[], ldb
                      c_int  # batchCount
                      ]

# cublasStatus_t cublasChemm(cublasHandle_t handle,
#                            cublasSideMode_t side, cublasFillMode_t uplo,
#                            int m, int n,
#                            const cuComplex       *alpha,
#                            const cuComplex       *A, int lda,
#                            const cuComplex       *B, int ldb,
#                            const cuComplex       *beta,
#                            cuComplex       *C, int ldc)
# cublasStatus_t cublasZhemm(cublasHandle_t handle,
#                            cublasSideMode_t side, cublasFillMode_t uplo,
#                            int m, int n,
#                            const cuDoubleComplex *alpha,
#                            const cuDoubleComplex *A, int lda,
#                            const cuDoubleComplex *B, int ldb,
#                            const cuDoubleComplex *beta,
#                            cuDoubleComplex *C, int ldc)
cublasChemm = libcublas.cublasChemm_v2
cublasZhemm = libcublas.cublasZhemm_v2
for funct in [cublasChemm, cublasZhemm]:
    funct.restype = cublasStatus_t
    funct.argtypes = [cublasHandle_t,
                      c_cublasSideMode_t,  # side
                      c_cublasFillMode_t,  # uplo
                      c_int, c_int,  # m, n
                      scalar_pointer,  # *alpha
                      memory_pointer, c_int,  # *A, lda
                      memory_pointer, c_int,  # *B, ldb
                      scalar_pointer,  # *beta
                      memory_pointer, c_int  # *C, ldc
                      ]

# cublasStatus_t cublasCherk(cublasHandle_t handle,
#                            cublasFillMode_t uplo, cublasOperation_t trans,
#                            int n, int k,
#                            const float  *alpha,
#                            const cuComplex       *A, int lda,
#                            const float  *beta,
#                            cuComplex       *C, int ldc)
# cublasStatus_t cublasZherk(cublasHandle_t handle,
#                            cublasFillMode_t uplo, cublasOperation_t trans,
#                            int n, int k,
#                            const double *alpha,
#                            const cuDoubleComplex *A, int lda,
#                            const double *beta,
#                            cuDoubleComplex *C, int ldc)
cublasCherk = libcublas.cublasCherk_v2
cublasZherk = libcublas.cublasZherk_v2
for funct in [cublasCherk, cublasZherk]:
    funct.restype = cublasStatus_t
    funct.argtypes = [cublasHandle_t,
                      c_cublasFillMode_t,  # uplo
                      c_cublasOperation_t,  # trans
                      c_int, c_int,  # n, k
                      scalar_pointer,  # *alpha
                      memory_pointer, c_int,  # *A, lda
                      scalar_pointer,  # *beta
                      memory_pointer, c_int  # *C, ldc
                      ]

# cublasStatus_t cublasCher2k(cublasHandle_t handle,
#                             cublasFillMode_t uplo, cublasOperation_t trans,
#                             int n, int k,
#                             const cuComplex       *alpha,
#                             const cuComplex       *A, int lda,
#                             const cuComplex       *B, int ldb,
#                             const float  *beta,
#                             cuComplex       *C, int ldc)
# cublasStatus_t cublasZher2k(cublasHandle_t handle,
#                             cublasFillMode_t uplo, cublasOperation_t trans,
#                             int n, int k,
#                             const cuDoubleComplex *alpha,
#                             const cuDoubleComplex *A, int lda,
#                             const cuDoubleComplex *B, int ldb,
#                             const double *beta,
#                             cuDoubleComplex *C, int ldc)
cublasCher2k = libcublas.cublasCher2k_v2
cublasZher2k = libcublas.cublasZher2k_v2
for funct in [cublasCher2k, cublasZher2k]:
    funct.restype = cublasStatus_t
    funct.argtypes = [cublasHandle_t,
                      c_cublasFillMode_t,  # uplo
                      c_cublasOperation_t,  # trans
                      c_int, c_int,  # n, k
                      scalar_pointer,  # *alpha
                      memory_pointer, c_int,  # *A, lda
                      memory_pointer, c_int,  # *B, ldb
                      scalar_pointer,  # *beta
                      memory_pointer, c_int  # *C, ldc
                      ]

# cublasStatus_t cublasCherkx(cublasHandle_t handle,
#                             cublasFillMode_t uplo, cublasOperation_t trans,
#                             int n, int k,
#                             const cuComplex       *alpha,
#                             const cuComplex       *A, int lda,
#                             const cuComplex       *B, int ldb,
#                             const float  *beta,
#                             cuComplex       *C, int ldc)
# cublasStatus_t cublasZherkx(cublasHandle_t handle,
#                             cublasFillMode_t uplo, cublasOperation_t trans,
#                             int n, int k,
#                             const cuDoubleComplex *alpha,
#                             const cuDoubleComplex *A, int lda,
#                             const cuDoubleComplex *B, int ldb,
#                             const double *beta,
#                             cuDoubleComplex *C, int ldc)
cublasCherkx = libcublas.cublasCherkx
cublasZherkx = libcublas.cublasZherkx
for funct in [cublasCherkx, cublasZherkx]:
    funct.restype = cublasStatus_t
    funct.argtypes = [cublasHandle_t,
                      c_cublasFillMode_t,  # uplo
                      c_cublasOperation_t,  # trans
                      c_int, c_int,  # n, k
                      scalar_pointer,  # *alpha
                      memory_pointer, c_int,  # *A, lda
                      memory_pointer, c_int,  # *B, ldb
                      scalar_pointer,  # *beta
                      memory_pointer, c_int  # *C, ldc
                      ]

## cuBLAS BLAS-like Extension Functions ##

# cublasStatus_t cublasSgeam(cublasHandle_t handle,
#                           cublasOperation_t transa, cublasOperation_t transb,
#                           int m, int n,
#                           const float *alpha,
#                           const float *A, int lda,
#                           const float *beta,
#                           const float *B, int ldb,
#                           float *C, int ldc)
# cublasStatus_t cublasDgeam(cublasHandle_t handle,
#                           cublasOperation_t transa, cublasOperation_t transb,
#                           int m, int n,
#                           const double *alpha,
#                           const double *A, int lda,
#                           const double *beta,
#                           const double *B, int ldb,
#                           double *C, int ldc)
# cublasStatus_t cublasCgeam(cublasHandle_t handle,
#                           cublasOperation_t transa, cublasOperation_t transb,
#                           int m, int n,
#                           const cuComplex *alpha,
#                           const cuComplex *A, int lda,
#                           const cuComplex *beta ,
#                           const cuComplex *B, int ldb,
#                           cuComplex *C, int ldc)
# cublasStatus_t cublasZgeam(cublasHandle_t handle,
#                           cublasOperation_t transa, cublasOperation_t transb,
#                           int m, int n,
#                           const cuDoubleComplex *alpha,
#                           const cuDoubleComplex *A, int lda,
#                           const cuDoubleComplex *beta,
#                           const cuDoubleComplex *B, int ldb,
#                           cuDoubleComplex *C, int ldc)
cublasSgeam = libcublas.cublasSgeam
cublasDgeam = libcublas.cublasDgeam
cublasCgeam = libcublas.cublasCgeam
cublasZgeam = libcublas.cublasZgeam
for funct in [cublasSgeam, cublasDgeam, cublasCgeam, cublasZgeam]:
    funct.restype = cublasStatus_t
    funct.argtypes = [cublasHandle_t,
                      c_cublasOperation_t,  # trans
                      c_cublasOperation_t,  # transb
                      c_int, c_int,  # m, n
                      scalar_pointer,  # *alpha
                      memory_pointer, c_int,  # *A, lda
                      scalar_pointer,  # *beta
                      memory_pointer, c_int,  # *B, ldb
                      memory_pointer, c_int  # *C, ldc
                      ]

# cublasStatus_t cublasSdgmm(cublasHandle_t handle, cublasSideMode_t mode,
#                           int m, int n,
#                           const float *A, int lda,
#                           const float *x, int incx,
#                           float *C, int ldc)
# cublasStatus_t cublasDdgmm(cublasHandle_t handle, cublasSideMode_t mode,
#                           int m, int n,
#                           const double *A, int lda,
#                           const double *x, int incx,
#                           double *C, int ldc)
# cublasStatus_t cublasCdgmm(cublasHandle_t handle, cublasSideMode_t mode,
#                           int m, int n,
#                           const cuComplex *A, int lda,
#                           const cuComplex *x, int incx,
#                           cuComplex *C, int ldc)
# cublasStatus_t cublasZdgmm(cublasHandle_t handle, cublasSideMode_t mode,
#                           int m, int n,
#                           const cuDoubleComplex *A, int lda,
#                           const cuDoubleComplex *x, int incx,
#                           cuDoubleComplex *C, int ldc)
cublasSdgmm = libcublas.cublasSdgmm
cublasDdgmm = libcublas.cublasDdgmm
cublasCdgmm = libcublas.cublasCdgmm
cublasZdgmm = libcublas.cublasZdgmm
for funct in [cublasSdgmm, cublasDdgmm, cublasCdgmm, cublasZdgmm]:
    funct.restype = cublasStatus_t
    funct.argtypes = [cublasHandle_t,
                      c_cublasSideMode_t,  # mode
                      c_int, c_int,  # m, n
                      memory_pointer, c_int,  # *A, lda
                      memory_pointer, c_int,  # *x, incx
                      memory_pointer, c_int  # *C, ldc
                      ]


# cublasStatus_t cublasSgetrfBatched(cublasHandle_t handle,
#                                   int n,
#                                   float *Aarray[],
#                                   int lda,
#                                   int *PivotArray,
#                                   int *infoArray,
#                                   int batchSize);
# cublasStatus_t cublasDgetrfBatched(cublasHandle_t handle,
#                                   int n,
#                                   double *Aarray[],
#                                   int lda,
#                                   int *PivotArray,
#                                   int *infoArray,
#                                   int batchSize);
# cublasStatus_t cublasCgetrfBatched(cublasHandle_t handle,
#                                   int n,
#                                   cuComplex *Aarray[],
#                                   int lda,
#                                   int *PivotArray,
#                                   int *infoArray,
#                                   int batchSize);
# cublasStatus_t cublasZgetrfBatched(cublasHandle_t handle,
#                                   int n,
#                                   cuDoubleComplex *Aarray[],
#                                   int lda,
#                                   int *PivotArray,
#                                   int *infoArray,
#                                   int batchSize);
cublasSgetrfBatched = libcublas.cublasSgetrfBatched
cublasDgetrfBatched = libcublas.cublasDgetrfBatched
cublasCgetrfBatched = libcublas.cublasCgetrfBatched
cublasZgetrfBatched = libcublas.cublasZgetrfBatched
for funct in [cublasSgetrfBatched, cublasDgetrfBatched, cublasCgetrfBatched, cublasZgetrfBatched]:
    funct.restype = cublasStatus_t
    funct.argtypes = [cublasHandle_t,
                      c_int,  # n
                      array_pointer,  # *Aarray[]
                      c_int,  # lda
                      scalar_pointer,  # *PivotArray
                      scalar_pointer,  # *infoArray
                      c_int  # batchSize
                      ]

# cublasStatus_t cublasSgetrsBatched(cublasHandle_t handle,
#                                   cublasOperation_t trans,
#                                   int n,
#                                   int nrhs,
#                                   const float *Aarray[],
#                                   int lda,
#                                   const int *devIpiv,
#                                   float *Barray[],
#                                   int ldb,
#                                   int *info,
#                                   int batchSize);
# cublasStatus_t cublasDgetrsBatched(cublasHandle_t handle,
#                                   cublasOperation_t trans,
#                                   int n,
#                                   int nrhs,
#                                   const double *Aarray[],
#                                   int lda,
#                                   const int *devIpiv,
#                                   double *Barray[],
#                                   int ldb,
#                                   int *info,
#                                   int batchSize);
# cublasStatus_t cublasCgetrsBatched(cublasHandle_t handle,
#                                   cublasOperation_t trans,
#                                   int n,
#                                   int nrhs,
#                                   const cuComplex *Aarray[],
#                                   int lda,
#                                   const int *devIpiv,
#                                   cuComplex *Barray[],
#                                   int ldb,
#                                   int *info,
#                                   int batchSize);
# cublasStatus_t cublasZgetrsBatched(cublasHandle_t handle,
#                                   cublasOperation_t trans,
#                                   int n,
#                                   int nrhs,
#                                   const cuDoubleComplex *Aarray[],
#                                   int lda,
#                                   const int *devIpiv,
#                                   cuDoubleComplex *Barray[],
#                                   int ldb,
#                                   int *info,
#                                   int batchSize);
cublasSgetrsBatched = libcublas.cublasSgetrsBatched
cublasDgetrsBatched = libcublas.cublasDgetrsBatched
cublasCgetrsBatched = libcublas.cublasCgetrsBatched
cublasZgetrsBatched = libcublas.cublasZgetrsBatched
for funct in [cublasSgetrsBatched, cublasDgetrsBatched, cublasCgetrsBatched, cublasZgetrsBatched]:
    funct.restype = cublasStatus_t
    funct.argtypes = [cublasHandle_t,
                      c_cublasOperation_t,  # trans
                      c_int,  # n
                      c_int,  # nrhs
                      array_pointer,  # *Aarray[]
                      c_int,  # lda
                      scalar_pointer,  # *devIpiv
                      array_pointer,  # *Barray[]
                      c_int,  # ldb
                      scalar_pointer,  # *info
                      c_int  # batchSize
                      ]

# cublasStatus_t cublasSgetriBatched(cublasHandle_t handle,
#                                   int n,
#                                   float *Aarray[],
#                                   int lda,
#                                   int *PivotArray,
#                                   float *Carray[],
#                                   int ldc,
#                                   int *infoArray,
#                                   int batchSize);
# cublasStatus_t cublasDgetriBatched(cublasHandle_t handle,
#                                   int n,
#                                   double *Aarray[],
#                                   int lda,
#                                   int *PivotArray,
#                                   double *Carray[],
#                                   int ldc,
#                                   int *infoArray,
#                                   int batchSize);
# cublasStatus_t cublasCgetriBatched(cublasHandle_t handle,
#                                   int n,
#                                   cuComplex *Aarray[],
#                                   int lda,
#                                   int *PivotArray,
#                                   cuComplex *Carray[],
#                                   int ldc,
#                                   int *infoArray,
#                                   int batchSize);
# cublasStatus_t cublasZgetriBatched(cublasHandle_t handle,
#                                   int n,
#                                   cuDoubleComplex *Aarray[],
#                                   int lda,
#                                   int *PivotArray,
#                                   cuDoubleComplex *Carray[],
#                                   int ldc,
#                                   int *infoArray,
#                                   int batchSize);
cublasSgetriBatched = libcublas.cublasSgetriBatched
cublasDgetriBatched = libcublas.cublasDgetriBatched
cublasCgetriBatched = libcublas.cublasCgetriBatched
cublasZgetriBatched = libcublas.cublasZgetriBatched
for funct in [cublasSgetriBatched, cublasDgetriBatched, cublasCgetriBatched, cublasZgetriBatched]:
    funct.restype = cublasStatus_t
    funct.argtypes = [cublasHandle_t,
                      c_int,  # n
                      array_pointer,  # *Aarray[]
                      c_int,  # lda
                      scalar_pointer,  # *PivotArray,
                      array_pointer,  # *Carray[]
                      c_int,  # ldc
                      scalar_pointer,  # *infoArray
                      c_int  # batchSize
                      ]


# Published symbols in libcublas.so.7.0 not implemented yet:

# cublasXerbla          ** Not documented??
# cublasSetBackdoor     ** Not documented??

# cublasGetStream_v2
# cublasSetStream_v2
# cublasGetVectorAsync
# cublasSetVectorAsync
# cublasGetMatrixAsync
# cublasSetMatrixAsync

# cublasAlloc           ** Deprecated, use cudaMalloc
# cublasFree            ** Deprecated, use cudaFree
# cublasInit            ** Deprecated, use cublasCreate
# cublasShutdown        ** Deprecated, use cublasDestroy
# cublasGetError        ** Deprecated, use functions return value
# cublasSetKernelStream ** Deprecated, use cublasSetStream_v2


# cublasCgetri ** Not documented??
# cublasDgetri ** Not documented??
# cublasSgetri ** Not documented??
# cublasZgetri ** Not documented??

# cublasCbdmm  ** Not documented??
# cublasDbdmm  ** Not documented??
# cublasSbdmm  ** Not documented??
# cublasZbdmm  ** Not documented??


# cublasCmatinvBatched
# cublasDmatinvBatched
# cublasSmatinvBatched
# cublasZmatinvBatched

# cublasCgeqrfBatched
# cublasDgeqrfBatched
# cublasSgeqrfBatched
# cublasZgeqrfBatched

# cublasCgelsBatched
# cublasDgelsBatched
# cublasSgelsBatched
# cublasZgelsBatched

# cublasCtrttp
# cublasDtrttp
# cublasStrttp
# cublasZtrttp

# cublasCtpttr
# cublasDtpttr
# cublasStpttr
# cublasZtpttr
