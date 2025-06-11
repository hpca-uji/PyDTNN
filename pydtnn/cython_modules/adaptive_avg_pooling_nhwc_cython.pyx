import numpy as np
cimport numpy as np
cimport cython
from cython.parallel import prange

# --- COMMON --- #
ctypedef fused npDT:
    np.int8_t
    np.float32_t
    np.float64_t
    # NOTE: in order to extend the supported data types, add the new types here.
# -- END npDT -- #

@cython.cdivision(True)
cdef inline int index_first_element(int index, int dim_in, int dim_out) nogil:
    return ((index * dim_in) / dim_out)
# --- END index_first_element --- #

@cython.cdivision(True)
cdef inline int index_last_element(int index, int dim_in, int dim_out) nogil:
    return ((((index + 1) * dim_in) + dim_out - 1) / dim_out)
# --- END index_last_element --- #

# --- END COMMON --- #

# =================== #
# =================== #

# --- FORWARD --- #

def adaptive_avg_pooling_fwd_nhwc_cython(np.ndarray x, int new_h, int new_w) -> np.ndarray:
    cdef int n = x.shape[0]
    cdef int h = x.shape[1]
    cdef int w = x.shape[2]
    cdef int c = x.shape[3]        

    cdef np.ndarray pooled_x = np.zeros((n, new_h, new_w, c), dtype = x.dtype)

    try:
        avg_pooling(pooled_x, x, n, c, h, w, new_h, new_w)
        return pooled_x
    except TypeError as e:
        raise TypeError(f"Function: \"adaptive_avg_pooling_fwd_nhwc_cython\". Error: {e}")
# --- END adaptive_avg_pooling_fwd_nchw_cython --- #

def avg_pooling(np.ndarray[npDT, ndim=4] pooled_x, 
                 np.ndarray[npDT, ndim=4] x, 
                 int n, int c, int h, int w, int new_h, int new_w):

    cdef npDT[:,:,:,:] pooled_x_view = pooled_x    
    cdef const npDT[:,:,:,:] x_view = x
    _avg_pooling(pooled_x_view, x_view, n, c, h, w, new_h, new_w)
# --- END avg_pooling --- #

@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
# NOTE: "npDT[:, :, :, :]" this is a view of a 4 dimensions array-like object of one of the supported types.
cdef _avg_pooling(npDT[:, :, :, :] pooled_x,
                  const npDT[:, :, :, :] x,
                  int n, int c, int h, int w,
                  int new_h, int new_w):
                  
    cdef int h_start, h_end, w_start, w_end
    cdef int nn, cc, hi, wi, i, j
    cdef int elements_h, elements
    cdef npDT add

    for nn in prange(n, nogil=True):
        for cc in range(c):
            for hi in range(new_h):
                h_start = index_first_element(hi, h, new_h)
                h_end = index_last_element(hi, h, new_h)
                elements_h = h_end - h_start            
                for wi in range(new_w):
                    w_start = index_first_element(wi, w, new_w)
                    w_end = index_last_element(wi, w, new_w)
                    elements = elements_h * (w_end - w_start)
    
                    add = <npDT> 0.0
                    for i in range(h_start, h_end):
                        for j in range(w_start, w_end):
                            # If it is not done in this way (e.g.: add += x[nn, cc, i, j]),
                            #   Cython thinks that "add" is shared for all the threads and throws an error.
                            add = add + x[nn, i, j, cc]

                    pooled_x[nn, hi, wi, cc] = add / elements
# --- END _avg_pooling --- #

# --- END FORWARD --- #

# =================== #
# =================== #

# --- BACKWARD --- #

def adaptive_avg_pooling_bwd_nhwc_cython(np.ndarray dy, int new_h, int new_w) -> np.ndarray:
    cdef int n = dy.shape[0]
    cdef int h = dy.shape[1]
    cdef int w = dy.shape[2]
    cdef int c = dy.shape[3]        

    cdef np.ndarray dx = np.zeros((n, new_h, new_w, c), dtype = dy.dtype)

    try:
        backward_avg_pooling(dx, dy, n, c, h, w, new_h, new_w)
        return dx
    except TypeError as e:
        raise TypeError(f"Function: \"adaptive_avg_pooling_bwd_nhwc_cython\". Error: {e}")
# --- END adaptive_avg_pooling_bwd_nhwc_cython --- #

def backward_avg_pooling(np.ndarray[npDT, ndim=4] dx, 
                         np.ndarray[npDT, ndim=4] dy, 
                         int n, int c, int h, int w, int new_h, int new_w):

    cdef npDT[:,:,:,:] y_view = dx
    cdef const npDT[:,:,:,:] x_view = dy
    _backward_avg_pooling(y_view, x_view, n, c, h, w, new_h, new_w)
# --- END avg_pooling --- #

@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
# NOTE: "npDT[:, :, :, :]" this is a view of a 4 dimensions array-like object of one of the supported types.
cdef _backward_avg_pooling(npDT[:, :, :, :] dx,
                  const npDT[:, :, :, :] dy,
                  int n, int c, int h, int w,
                  int new_h, int new_w):
                  
    cdef int h_start, h_end, w_start, w_end
    cdef int nn, cc, ho, wo, i, j
    cdef int elements_h, elements
    cdef npDT delta

    for nn in prange(n, nogil=True):
        for cc in range(c):
            for ho in range(new_h):
                h_start = index_first_element(ho, h, new_h)
                h_end = index_last_element(ho, h, new_h)
                elements_h = h_end - h_start            
                for wo in range(new_w):
                    w_start = index_first_element(wo, w, new_w)
                    w_end = index_last_element(wo, w, new_w)
                    elements = elements_h * (w_end - w_start)

                    delta = dy[nn, ho, wo, cc] / elements
                    for i in range(h_start, h_end):
                        for j in range(w_start, w_end):
                                dx[nn, i, j, cc] += delta
# --- END _backward_avg_pooling --- #

# --- END BACKWARD --- #
