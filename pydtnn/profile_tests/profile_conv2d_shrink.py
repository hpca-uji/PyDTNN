"""
Performance tests for shrinking matrices

For running the tests run:
    python profile_tests/profile_conv2d_shrink.py

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
    from NN_pad_cython import shrink_cython
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


def pad_numpy(vpadding, hpadding, matrix_in, matrix_out):
    h, w = matrix_out.shape[2:4]
    matrix_out[...] = matrix_in[:, :, vpadding:vpadding + h, hpadding:hpadding + w]


def time_shrink(x_shape, vpadding, hpadding, dtype=np.float32):
    b, c, h, w = x_shape
    new_h = h + 2 * vpadding
    new_w = w + 2 * hpadding
    matrix_in = np.random.rand(b, c, new_h, new_w).astype(dtype=dtype)
    numpy_matrix_out = np.zeros((b, c, h, w), dtype=dtype, order="C")
    cython_matrix_out = np.zeros((b, c, h, w), dtype=dtype, order="C")
    #
    # First run
    #
    print(f"First pass {x_shape} (checking outputs)", sep="", end="")
    #
    print(".", sep="", end="")
    pad_numpy(vpadding, hpadding, matrix_in, numpy_matrix_out)
    print(".", sep="", end="")
    shrink_cython(matrix_in, cython_matrix_out)
    try:
        assert np.allclose(numpy_matrix_out, cython_matrix_out), "numpy and cython version differ"
    except AssertionError as err:
        print()
        print(err)
        print("Numpy pad:", numpy_matrix_out.shape)
        print(numpy_matrix_out)
        print("Cython pad:", cython_matrix_out.shape)
        print(cython_matrix_out)
    #
    # Second run
    #
    print(f"Second pass {x_shape} (getting times)", sep="", end="")
    print(".", sep="", end="")
    numpy_shrink_t = timeit(lambda: pad_numpy(vpadding, hpadding, matrix_in, numpy_matrix_out),
                            number=10) / 10
    print(".", sep="", end="")
    cython_shrink_t = timeit(lambda: shrink_cython(matrix_in, cython_matrix_out),
                             number=10) / 10
    print()
    min_t = np.min([numpy_shrink_t, cython_shrink_t])
    a = "*" if cython_shrink_t == min_t else ""
    return [["numpy", "a", "cython"],
            ["{:6.4f}".format(numpy_shrink_t),
             a,
             "{:6.4f}".format(cython_shrink_t - numpy_shrink_t),
             ]]


if __name__ == '__main__':
    # D(b, c, h, w, kn, kh, kw, vpadding, hpadding, vstride, hstride):
    layers = [
        D(1, 1, 3, 3, 64, 3, 3, 1, 1, 2, 2),
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
        # w <- y (kn * b * ho * wo)
        backward_layers.append(D(layer.c, layer.b, layer.h, layer.w, layer.kn, layer.ho, layer.wo,
                                 layer.vpadding, layer.hpadding, layer.vstride, layer.hstride))
    layers += backward_layers
    t = None
    for layer in layers:
        _x_shape = (layer.b, layer.c, layer.h, layer.w)
        headers, values = time_shrink(_x_shape, layer.vpadding, layer.hpadding)
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
