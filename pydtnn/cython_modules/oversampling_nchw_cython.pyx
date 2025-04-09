#
#  This file is part of Python Distributed Training of Neural Networks (PyDTNN)
#
#  Copyright (C) 2025 Universitat Jaume I
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

ctypedef fused supported_types_t:
    np.int8_t
    np.float32_t
    np.float64_t
    # NOTE: in order to extend the supported data types, add the new types here.
# -- END supported_types_t -- #

# NOTE: "supported_types_t[:, :, :, :]" is a view of a 4 dimensions array-like object of one of the supported types.
cdef _oversampling(supported_types_t[:, :, :, :] oversampled_x, 
                  const supported_types_t[:, :, :, :] x, 
                  int n, int c, int h, int w, int extra_h, int extra_w):

    cdef int row, column
    cdef int nn, cc, hi, hj, i, j

    for nn in prange(n, nogil=True):
        for cc in range(c):
            for hi in range(h):
                for hj in range(w):
                    for i in range(extra_h):
                        for j in range(extra_w):
                            row = i + (hi * extra_h)
                            column = j + (hj * extra_w)
                            oversampled_x[nn, cc, row, column] = x[nn, cc, hi, hj]
# --- END _oversampling --- #

# This intermediate function "selects" the data type among the supported types
@cython.boundscheck(False)
@cython.wraparound(False)
def oversampling(np.ndarray[supported_types_t, ndim=4] oversampled_x, 
                 np.ndarray[supported_types_t, ndim=4] x, 
                 int n, int c, int h, int w, int extra_h, int extra_w):

    cdef supported_types_t[:,:,:,:] oversampled_x_view = oversampled_x    
    cdef const supported_types_t[:,:,:,:] x_view = x
    _oversampling(oversampled_x_view, x_view, n, c, h, w, extra_h, extra_w)
# --- END oversampling --- #

@cython.boundscheck(False)
@cython.wraparound(False)
# extra_h = new_h // h, new_h >= h ; extra_w = new_w // w, new_w >= w;
def oversampling_fwd_nchw_cython(np.ndarray x, int new_h, int new_w, 
                                int extra_h, int extra_w) -> np.ndarray:
    cdef int n = x.shape[0]
    cdef int c = x.shape[1]
    cdef int h = x.shape[2]
    cdef int w = x.shape[3]

    cdef np.ndarray oversampled_x = np.empty((n, c, new_h, new_w), dtype = x.dtype)

    try:
        oversampling(oversampled_x, x, n, c, h, w, extra_h, extra_w)
        return oversampled_x
    except TypeError:
        raise TypeError(f"Type '{x.dtype}' is not supported by oversampling_fwd_nchw_cython!")
# --- END oversampling_fwd_nchw_cython --- #

###########################################################

###########################################################

###########################################################

# Version with different functions for every type of data.
# NOTE: The performance of this version is a little bit better than the previous one, but it's necessary to make a function for every supported type while the other is generic.
# This version: 8,998s; previous one: 9,396s | Test: n = 6000, c = 100, h_size = 7, w_size = 7, out_h = 100, out_w = 100

@cython.boundscheck(False)
@cython.wraparound(False)
def _oversampling_int_8(np.ndarray[np.int8_t, ndim=4] oversampled_x,
                            np.ndarray[np.int8_t, ndim=4] x,
                            int n, int c, int h, int w, int extra_h, int extra_w) -> np.ndarray:
    cdef int row, column
    cdef int nn, cc, hi, hj, i, j

    for nn in prange(n, nogil=True):
        for cc in range(c):
            for hi in range(h):
                for hj in range(w):
                    for i in range(extra_h):
                        for j in range(extra_w):
                            row = i + (hi * extra_h)
                            column = j + (hj * extra_w)
                            oversampled_x[nn, cc, row, column] = x[nn, cc, hi, hj]
# --- END _oversampling_float_8 --- #

@cython.boundscheck(False)
@cython.wraparound(False)
def _oversampling_float_32(np.ndarray[np.float32_t, ndim=4] oversampled_x,
                            np.ndarray[np.float32_t, ndim=4] x,
                            int n, int c, int h, int w, int extra_h, int extra_w) -> np.ndarray:
    cdef int row, column
    cdef int nn, cc, hi, hj, i, j
    
    for nn in prange(n, nogil=True):
        for cc in range(c):
            for hi in range(h):
                for hj in range(w):
                    for i in range(extra_h):
                        for j in range(extra_w):
                            row = i + (hi * extra_h)
                            column = j + (hj * extra_w)
                            oversampled_x[nn, cc, row, column] = x[nn, cc, hi, hj]
# --- END _oversampling_float_32 --- #

@cython.boundscheck(False)
@cython.wraparound(False)
def _oversampling_float_64(np.ndarray[np.float64_t, ndim=4] oversampled_x,
                            np.ndarray[np.float64_t, ndim=4] x,
                            int n, int c, int h, int w, int extra_h, int extra_w) -> np.ndarray:
    cdef int row, column
    cdef int nn, cc, hi, hj, i, j
    
    for nn in prange(n, nogil=True):
        for cc in range(c):
            for hi in range(h):
                for hj in range(w):
                    for i in range(extra_h):
                        for j in range(extra_w):
                            row = i + (hi * extra_h)
                            column = j + (hj * extra_w)
                            oversampled_x[nn, cc, row, column] = x[nn, cc, hi, hj]
# --- END _oversampling_float_64 --- #

@cython.boundscheck(False)
@cython.wraparound(False)
# extra_h = new_h // h, new_h >= h ; extra_w = new_w // w, new_w >= w;
def _oversampling_fwd_nchw_cython(np.ndarray x, int new_h, int new_w, 
                                int extra_h, int extra_w) -> np.ndarray:
    cdef int n = x.shape[0]
    cdef int c = x.shape[1]
    cdef int h = x.shape[2]
    cdef int w = x.shape[3]
    cdef dict pseudo_switch = {    
        np.dtype('int8'): _oversampling_int_8,
        np.dtype('float32'): _oversampling_float_32,
        np.dtype('float64'): _oversampling_float_64,
    }

    cdef np.ndarray oversampled_x = np.empty((n, c, new_h, new_w), dtype = x.dtype)

    if x.dtype in pseudo_switch:
        pseudo_switch[x.dtype](oversampled_x, x, n, c, h, w, extra_h, extra_w)
        return oversampled_x
    else: 
        raise TypeError(f" Type received '{x.dtype}'. Types expected in oversampling_fwd_nchw_cython: {pseudo_switch.keys()}")
# -- END oversampling_nchw_cython -- #

###########################################################
###########################################################
###########################################################

# This version below is the slower due the use of GIL.
# TODO: Check if it is possible to remove the uses of GIL.

@cython.boundscheck(False)
@cython.wraparound(False)
# extra_h = new_h // h, new_h >= h ; extra_w = new_w // w, new_w >= w;
def _oversampling_nchw_cython(np.ndarray x, int new_h, int new_w, 
                                int extra_h, int extra_w) -> np.ndarray:
    cdef int n = x.shape[0]
    cdef int c = x.shape[1]
    cdef int h = x.shape[2]
    cdef int w = x.shape[3]

    cdef np.ndarray oversampled_x = np.empty((n, c, new_h, new_w), dtype = x.dtype)

    
    return __oversampling_float_32(oversampled_x=oversampled_x, x=x, 
                                    n=n, c=c, h=h, w=w, 
                                    extra_h=extra_h, extra_w=extra_w)
#--- END oversampling_nchw_cython ---#

@cython.boundscheck(False)
@cython.wraparound(False)
def __oversampling_float_32(np.ndarray[np.float32_t, ndim=4] oversampled_x,
                            np.ndarray[np.float32_t, ndim=4] x,
                            int n, int c, int h, int w, 
                            int extra_h, int extra_w) -> np.ndarray:
    
    cdef int nn, cc, i, j
    
    cdef np.ndarray to_concat_h = np.empty(h, dtype=object)
    cdef np.ndarray to_concat_w = np.empty(w, dtype=object)

    for nn in prange(n, nogil=True):
        for cc in range(c):
            for i in range(h):
                for j in range(w):
            # TODO: check if it is possible to remove this 3 uses of the GIL.
                    with gil:
                        to_concat_w[j] = np.full((extra_h, extra_w), x[nn, cc, i, j], dtype=x.dtype)
                with gil:
                    to_concat_h[i] = np.concatenate(to_concat_w, axis=1)
            with gil:  
                oversampled_x[nn, cc] = np.concatenate(to_concat_h, axis=0)
    
    return np.block(oversampled_x)
#--- END _oversampling_float_32 ---#