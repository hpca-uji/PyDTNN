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

import numpy as np
cimport numpy as np
cimport cython
from cython.parallel import prange

ctypedef fused npDT:
    np.int8_t
    np.float32_t
    np.float64_t

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
def transpose_0231_ikj_cython(np.ndarray[npDT, ndim=4] original,
                              np.ndarray[npDT, ndim=4] transposed) -> None:
    """
    Transposes a 4D matrix from (0,1,2,3) to (0,2,3,1).
    This is equivalent to transpose a 3D matrix 0x1x2·3 to 0x2·3x1
    """
    cdef npDT[:,:,:] orig3d = original.reshape((original.shape[0], original.shape[1], -1), copy=False)
    cdef npDT[:,:,:] trans3d = transposed.reshape((transposed.shape[0], -1, transposed.shape[3]), copy=False)

    cdef Py_ssize_t d0, d1, d2
    for d0 in prange(orig3d.shape[0], nogil=True, schedule="static"):
        for d2 in range(orig3d.shape[2]):
            for d1 in range(orig3d.shape[1]):
                trans3d[d0, d2, d1] = orig3d[d0, d1, d2]
# --- END transpose_0231_ikj_cython --- #

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
def transpose_0231_ijk_cython(np.ndarray[npDT, ndim=4] original,
                              np.ndarray[npDT, ndim=4] transposed) -> None:
    """
    Transposes a 4D matrix from (0,1,2,3) to (0,2,3,1).
    This is equivalent to transpose a 3D matrix 0x1x2·3 to 0x2·3x1
    """
    cdef npDT[:,:,:] orig3d = original.reshape((original.shape[0], original.shape[1], -1), copy=False)
    cdef npDT[:,:,:] trans3d = transposed.reshape((transposed.shape[0], -1, transposed.shape[3]), copy=False)

    cdef Py_ssize_t d0, d1, d2
    for d0 in prange(orig3d.shape[0], nogil=True, schedule="static"):
        for d1 in range(orig3d.shape[1]):
            for d2 in range(orig3d.shape[2]):
                trans3d[d0, d2, d1] = orig3d[d0, d1, d2]
# --- END transpose_0231_ijk_cython --- #

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
def transpose_0312_ikj_cython(np.ndarray[npDT, ndim=4] original,
                              np.ndarray[npDT, ndim=4] transposed) -> None:
    """
    Transposes a 4D matrix from (0,1,2,3) to (0,3,1,2).
    This is equivalent to transpose a 3D matrix 0x1·2x3 to 0x3x1·2
    """
    cdef npDT[:,:,:] orig3d = original.reshape((original.shape[0], -1, original.shape[3]), copy=False)
    cdef npDT[:,:,:] trans3d = transposed.reshape((transposed.shape[0], transposed.shape[1], -1), copy=False)

    cdef Py_ssize_t d0, d1, d2
    for d0 in prange(orig3d.shape[0], nogil=True, schedule="static"):
        for d2 in range(orig3d.shape[2]):
            for d1 in range(orig3d.shape[1]):
                trans3d[d0, d2, d1] = orig3d[d0, d1, d2]
# --- END transpose_0312_ikj_cython --- #

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
def transpose_0312_ijk_cython(np.ndarray[npDT, ndim=4] original,
                              np.ndarray[npDT, ndim=4] transposed) -> None:
    """
    Transposes a 4D matrix from (0,1,2,3) to (0,3,1,2).
    This is equivalent to transpose a 3D matrix 0x1·2x3 to 0x3x1·2
    """
    cdef npDT[:,:,:] orig3d = original.reshape((original.shape[0], -1, original.shape[3]), copy=False)
    cdef npDT[:,:,:] trans3d = transposed.reshape((transposed.shape[0], transposed.shape[1], -1), copy=False)

    cdef Py_ssize_t d0, d1, d2
    for d0 in prange(orig3d.shape[0], nogil=True, schedule="static"):
        for d1 in range(orig3d.shape[1]):
            for d2 in range(orig3d.shape[2]):
                trans3d[d0, d2, d1] = orig3d[d0, d1, d2]
# --- END transpose_0312_ijk_cython --- #

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
def transpose_1023_jik_cython(np.ndarray[npDT, ndim=4] original,
                              np.ndarray[npDT, ndim=4] transposed) -> None:
    """
    Transposes a 4D matrix from (0,1,2,3) to (1,0,2,3).
    This is equivalent to transpose a 3D matrix 0x1x2·3 to 1x0x2·3
    """
    cdef npDT[:,:,:] orig3d = original.reshape((original.shape[0], original.shape[1], -1), copy=False)
    cdef npDT[:,:,:] trans3d = transposed.reshape((transposed.shape[0], transposed.shape[1], -1), copy=False)

    cdef Py_ssize_t d0, d1, d2
    for d1 in prange(orig3d.shape[1], nogil=True, schedule="static"):
        for d0 in range(orig3d.shape[0]):
            for d2 in range(orig3d.shape[2]):
                trans3d[d1, d0, d2] = orig3d[d0, d1, d2]
# --- END transpose_1023_jik_cython --- #

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
def transpose_1023_ijk_cython(np.ndarray[npDT, ndim=4] original,
                              np.ndarray[npDT, ndim=4] transposed) -> None:
    """
    Transposes a 4D matrix from (0,1,2,3) to (0,2,3,1).
    This is equivalent to transpose a 3D matrix 0x1x2·3 to 1x0x2·3
    """
    cdef npDT[:,:,:] orig3d = original.reshape((original.shape[0], original.shape[1], -1), copy=False)
    cdef npDT[:,:,:] trans3d = transposed.reshape((transposed.shape[0], transposed.shape[1], -1), copy=False)

    cdef Py_ssize_t d0, d1, d2
    for d0 in prange(orig3d.shape[0], nogil=True, schedule="static"):
        for d1 in range(orig3d.shape[1]):
            for d2 in range(orig3d.shape[2]):
                trans3d[d1, d0, d2] = orig3d[d0, d1, d2]
# --- END transpose_1023_ijk_cython --- #
