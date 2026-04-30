cimport cython
from cython.parallel import prange
from pydtnn.backends.cython.utils.base cimport npDT

__all__ = (
    "adaptive_avg_pooling_fwd_nhwc_cython",
    "adaptive_avg_pooling_bwd_nhwc_cython"
)

@cython.cdivision(True)
cdef inline int index_first_element(int index, int dim_in, int dim_out) nogil:
    return ((index * dim_in) / dim_out)

@cython.cdivision(True)
cdef inline int index_last_element(int index, int dim_in, int dim_out) nogil:
    return ((((index + 1) * dim_in) + dim_out - 1) / dim_out)



# --- FORWARD ---

@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
def adaptive_avg_pooling_fwd_nhwc_cython(npDT[:,:,:,::1] x, 
                                         npDT[:,:,:,::1] pooled_x) -> None:
    cdef int n = x.shape[0]
    cdef int h = x.shape[1]
    cdef int w = x.shape[2]
    cdef int c = x.shape[3]

    cdef int new_h = pooled_x.shape[1]
    cdef int new_w = pooled_x.shape[2]

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



# --- BACKWARD ---

@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
def adaptive_avg_pooling_bwd_nhwc_cython(npDT[:,:,:,::1] dy, 
                                         npDT[:,:,:,::1] dx) -> None:
    cdef int n = dy.shape[0]
    cdef int h = dy.shape[1]
    cdef int w = dy.shape[2]
    cdef int c = dy.shape[3]

    cdef int new_h = dx.shape[1]
    cdef int new_w = dx.shape[2]

    cdef int h_start, h_end, w_start, w_end
    cdef int nn, cc, ho, wo, i, j
    cdef int elements_h, elements
    cdef npDT delta

    for nn in prange(n, nogil=True):
        for cc in range(c):
            for ho in range(h):
                h_start = index_first_element(ho, new_h, h)
                h_end = index_last_element(ho, new_h, h)
                elements_h = h_end - h_start
                for wo in range(w):
                    w_start = index_first_element(wo, new_w, w)
                    w_end = index_last_element(wo, new_w, w)
                    elements = elements_h * (w_end - w_start)

                    delta = dy[nn, ho, wo, cc] / elements
                    for i in range(h_start, h_end):
                        for j in range(w_start, w_end):
                            dx[nn, i, j, cc] += delta  

