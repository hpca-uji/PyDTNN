#
#  This file is part of Python Distributed Training of Neural Networks (PyDTNN)
#
#  Copyright (C) 2021-25 Universitat Jaume I
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

__all__ = (
    "relu_cython",
    "capped_relu_cython",
    "leaky_relu_cython"
)

import numpy as np
cimport numpy as np
cimport cython
from cython.parallel import prange

# Declare fused type npDT (to be used with template functions)
ctypedef fused npDT:
    np.int8_t
    np.float32_t
    np.float64_t

###############################################
#                 relu_cython                 #
###############################################

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
def relu_cython(npDT[::1] x, npDT[::1] max, np.int8_t[::1] mask) -> None:

    cdef int i
    for i in prange(x.shape[0], nogil=True):
        if x[i] > 0:
            max[i], mask[i] = x[i], 1
        else: 
            max[i], mask[i] = 0, 0
# --- END relu_cython --- #

###############################################
#             capped_relu_cython              #
###############################################

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
# NOTE: If cap = 6, then this is a Relu6.
def capped_relu_cython(npDT[::1] x, npDT[::1] max, np.int8_t[::1] mask, float cap) -> None:
    cdef int i
    for i in prange(x.shape[0], nogil=True):
        if x[i] >= cap:
            max[i], mask[i] = <npDT> cap, 1
        elif x[i] > 0: # cap > x[i] > 0
            max[i], mask[i] = x[i], 1
        else: #  x[i] <= 0
            max[i], mask[i] = <npDT> 0, 0
# --- END capped_relu_cython --- #

###############################################
#              leaky_relu_cython              #
###############################################

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
def leaky_relu_cython(npDT[::1] x, npDT[::1] max, npDT[::1] mask, float negative_slope) -> None:
    cdef int i
    for i in prange(x.shape[0], nogil=True):
        if x[i] > 0:
            max[i], mask[i] = x[i], 1
        elif x[i] < 0:
            max[i], mask[i] = <npDT> (x[i] * negative_slope), <npDT> negative_slope
        else: #x[i] == 0:
            max[i], mask[i] = 0, 0
# --- END leaky_relu_cython --- #
