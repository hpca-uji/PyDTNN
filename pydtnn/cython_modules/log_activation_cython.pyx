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

__all__ = (
    "log_fwd_cython",
    "log_bwd_cython"
)

import numpy as np
cimport numpy as np
cimport cython
from cython.parallel import prange
from cython.cimports.libc.math import exp, log

# Declare fused type npDT (to be used with template functions)
ctypedef fused npDT:
    np.int8_t
    np.float32_t
    np.float64_t

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
def log_fwd_cython(npDT[::1] x, npDT[::1] y) -> None:
    cdef int i
    # return np.log(1 / (1 + np.exp(-x)))
    

    for i in prange(x.shape[0], nogil=True):        
        y[i] = <npDT> log(1 / ( 1 + exp(-1*x[i])))
# --- END sigmoid_fwd_cython --- #

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
def log_bwd_cython(npDT[::1] dy, npDT[::1] dx) -> None:
    cdef int i
    # return 1 / (np.exp(dy) + 1)
    
    for i in prange(dy.shape[0], nogil=True):
        dx[i] = <npDT> (1 / (exp(dy[i]) + 1))
# --- END log_bwd_cython --- #

