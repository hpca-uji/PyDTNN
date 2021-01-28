"""
Performance tests for transposing matrices

For running the tests run:
    python profile_tests/cpr

"""
import inspect
import os
import platform
import sys
from timeit import timeit

import numpy as np
from rich import box
from rich.console import Console
from rich.table import Table

if True:
    current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
    parent_dir = os.path.dirname(current_dir)
    sys.path.insert(0, parent_dir)
    from NN_reindex_cython import reindex_cython
    from NN_layer import Conv2D


class D:

    def __init__(self, b, c, h, w, kn, kh, kw, vpadding, hpadding, vstride, hstride):
        self.b = b  # Batch size
        self.c = c  # Channels per layer
        self.h = h  # Layers height
        self.w = w  # Layers width
        self.kn = kn  # Number of filters
        self.kh = kh  # Filters weights height
        self.kw = kw  # Filters weights width
        self.vpadding = vpadding  # Vertical padding
        self.hpadding = hpadding  # Horizontal padding
        self.vstride = vstride  # Vertical stride
        self.hstride = hstride  # Horizontal stride

    @property
    def ho(self):
        return (self.h + 2 * self.vpadding - self.kh) // self.vstride + 1

    @property
    def wo(self):
        return (self.w + 2 * self.hpadding - self.kw) // self.hstride + 1

    def __repr__(self):
        return f"""\
x, weights, and y parameters:
  (b, c, h, w)    = {self.b} {self.c} {self.h} {self.w}
  (kn, c, kh, kw) = {self.kn} {self.c} {self.kh} {self.kw}
  (kn, b, ho, wo) = {self.kn} {self.b} {self.ho} {self.wo}
  padding         = {self.vpadding} {self.hpadding}
  stride          = {self.vstride} {self.hstride}
"""


class Params:
    pass


def reindex_numpy(h_new_indexes, v_new_indexes, matrix_in):
    matrix_out = matrix_in
    if h_new_indexes is not None:
        matrix_out = matrix_out[:, :, h_new_indexes, :]
    if v_new_indexes is not None:
        matrix_out = matrix_out[:, :, :, v_new_indexes]
    if h_new_indexes is not None or v_new_indexes is not None:
        # @warning: The copy() is required to ensure the correct order of the underlying data of
        #           matrix_out. Otherwise using self.cg_x_indexed.ravel(order="K") will lead to
        #           unexpected results
        matrix_out = matrix_out.copy()
    return matrix_out


def time_reindex(w_shape, x_shape, y_shape, vstride, hstride, dtype=np.float32):
    kn, c, kh, kw = w_shape
    b, c, h, w = x_shape
    kn, b, ho, wo = y_shape

    h_new_indexes, cg_vstride = Conv2D._get_x_new_indexes_and_xstride(kh, ho, vstride)
    v_new_indexes, cg_hstride = Conv2D._get_x_new_indexes_and_xstride(kw, wo, hstride)

    matrix_in = np.random.rand(*x_shape).astype(dtype=dtype)
    new_h = len(h_new_indexes) if h_new_indexes is not None else h
    new_w = len(v_new_indexes) if v_new_indexes is not None else w
    cython_matrix_out = np.empty((b, c, new_h, new_w), dtype=dtype, order="C")
    #
    # First run
    #
    print(f"First pass {x_shape} (checking outputs)", sep="", end="")
    #
    print(".", sep="", end="")
    numpy_matrix_out = reindex_numpy(h_new_indexes, v_new_indexes, matrix_in)
    print(".", sep="", end="")
    reindex_cython(h_new_indexes, v_new_indexes, matrix_in, cython_matrix_out)
    try:
        assert np.allclose(numpy_matrix_out, cython_matrix_out), "numpy and cython version differ"
    except AssertionError as err:
        print()
        print(err)
        print("Numpy reindex:", numpy_matrix_out.shape)
        print("Cython reindex:", cython_matrix_out.shape)
        print("np.where(numpy != cython):")
        print(np.where(numpy_matrix_out != cython_matrix_out))
    #
    # Second run
    #
    print(f"Second pass {x_shape} (getting times)", sep="", end="")
    print(".", sep="", end="")
    numpy_reindex_t = timeit(lambda: reindex_numpy(h_new_indexes, v_new_indexes, matrix_in),
                             number=10) / 10
    print(".", sep="", end="")
    cython_reindex_t = timeit(lambda: reindex_cython(h_new_indexes, v_new_indexes, matrix_in, cython_matrix_out),
                              number=10) / 10
    print()
    min_t = np.min([numpy_reindex_t, cython_reindex_t])
    a = "*" if cython_reindex_t == min_t else ""
    return [["numpy", "a", "cython"],
            ["{:6.4f}".format(numpy_reindex_t),
             a,
             "{:6.4f}".format(cython_reindex_t - numpy_reindex_t),
             ]]


if __name__ == '__main__':
    # D(b, c, h, w, kn, kh, kw, vpadding, hpadding, vstride, hstride):
    layers = [
        # AlexNet Cifar
        D(64, 3, 32, 32, 64, 3, 3, 1, 1, 2, 2),
        D(64, 64, 8, 8, 192, 3, 3, 1, 1, 1, 1),
        D(64, 192, 4, 4, 384, 3, 3, 1, 1, 1, 1),
        D(64, 384, 4, 4, 256, 3, 3, 1, 1, 1, 1),
        D(64, 256, 4, 4, 256, 3, 3, 1, 1, 1, 1),
        # AlexNet ImageNet
        D(64, 3, 227, 227, 96, 11, 11, 1, 1, 4, 4),
        D(64, 96, 27, 27, 256, 5, 5, 1, 1, 1, 1),
        D(64, 256, 13, 13, 384, 3, 3, 1, 1, 1, 1),
        D(64, 384, 13, 13, 384, 3, 3, 1, 1, 1, 1),
        D(64, 384, 13, 13, 256, 3, 3, 1, 1, 1, 1),
    ]
    backward_layers = []
    for layer in layers:
        if layer.vstride != 1 or layer.hstride != 1:
            continue
        # w <- y (kn * b * ho * wo)
        backward_layers.append(D(layer.c, layer.b, layer.h, layer.w, layer.kn, layer.ho, layer.wo,
                                 layer.vpadding, layer.hpadding, layer.vstride, layer.hstride))
    layers += backward_layers
    # Keep only those layers with vstride != 1 or hstride != 1
    layers = [la for la in layers if la.vstride != 1 or la.hstride != 1]
    # Increase x (h, w) dimensions by 2*vpadding, 2* hpadding
    for la in layers:
        la.h = la.h + 2 * la.vpadding
        la.vpadding = 0
        la.w = la.w + 2 * la.hpadding
        la.hpadding = 0
    t = None
    for layer in layers:
        _w_shape = (layer.kn, layer.c, layer.kh, layer.kw)
        _x_shape = (layer.b, layer.c, layer.h, layer.w)
        _y_shape = (layer.kn, layer.b, layer.ho, layer.wo)
        headers, values = time_reindex(_w_shape, _x_shape, _y_shape, layer.vstride, layer.hstride)
        if t is None:
            t = Table(box=box.HORIZONTALS, show_header=True, header_style="blue")
            t.add_column("x shape")
            for h in headers:
                t.add_column(str(h), justify="right")
        row_list = [", ".join([str(x) for x in _x_shape]), ] + values
        t.add_row(*row_list)
    print("*************************************************")
    print("** {}  OMP_NUM_THREADS: {}".format(platform.node(), os.environ["OMP_NUM_THREADS"]))
    print("** All times, except numpy, compared to numpy.")
    c = Console()
    c.print(t)
