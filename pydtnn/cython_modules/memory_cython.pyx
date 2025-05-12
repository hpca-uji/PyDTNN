"""Memory managment utilities"""

from libc.string cimport memchr
from cpython.buffer cimport PyObject_GetBuffer, PyBuffer_Release, PyBUF_CONTIG_RO

__all__ = (
    "memoryview_index",
)


def memoryview_index(view: memoryview, sub: bytes) -> int:
    """Find lowest index where substring is found"""
    cdef Py_buffer buffer
    cdef char* haystack
    cdef char* needle

    # Verification
    if len(sub) != 1:
        raise TypeError("Only single byte substring are supported")

    # Fast path
    if PyObject_GetBuffer(view, &buffer, PyBUF_CONTIG_RO) == 0:

        # Find substring
        try:
            haystack = <char*> buffer.buf
            needle = <char*> memchr(buffer.buf, sub[0], <size_t> buffer.len)
        finally:
            PyBuffer_Release(&buffer)

        # Compute index
        if needle:
            return <int>(needle - haystack)

    # Slow path
    else:
        for i, byte in enumerate(view):
            if byte == sub:
                return i

    raise ValueError("Substring not found")