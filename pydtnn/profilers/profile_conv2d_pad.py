"""
Performance tests for padding matrices

For running the tests run:
    python profile_conv2d_pad.py

"""

import os
import platform
from timeit import timeit

import numpy as np
from rich import box
from rich.console import Console
from rich.table import Table

from pydtnn.cython_modules import pad_cython


class D:

    def __init__(self, b, c, h, w, kn, kh, kw, vpadding, hpadding,
                 vstride, hstride, vdilation, hdilation):
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
        self.vdilation = vdilation  # Vertical dilation
        self.hdilation = hdilation  # Horizontal dilation

    @property
    def ho(self):
        return (self.h + 2 * self.vpadding - self.vdilation * (self.kh - 1) - 1) // self.vstride + 1

    @property
    def wo(self):
        return (self.w + 2 * self.hpadding - self.hdilation * (self.kw - 1) - 1) // self.hstride + 1

    def __repr__(self):
        return f"""\
x, weights, and y parameters:
  (b, c, h, w)    = {self.b} {self.c} {self.h} {self.w}
  (kn, c, kh, kw) = {self.kn} {self.c} {self.kh} {self.kw}
  (kn, b, ho, wo) = {self.kn} {self.b} {self.ho} {self.wo}
  padding         = {self.vpadding} {self.hpadding}
  stride          = {self.vstride} {self.hstride}
  dilation        = {self.vdilation} {self.hdilation}
"""


class Params:
    pass


def pad_numpy(vpadding, hpadding, matrix_in):
    matrix_out = np.pad(matrix_in,
                        ((0, 0), (0, 0),
                         (vpadding, vpadding), (hpadding, hpadding)),
                        mode='constant')
    return matrix_out


def time_pad(x_shape, vpadding, hpadding, dtype=np.float32):
    b, c, h, w = x_shape

    matrix_in = np.random.rand(*x_shape).astype(dtype=dtype)
    new_h = h + 2 * vpadding
    new_w = w + 2 * hpadding
    cython_matrix_out = np.empty((b, c, new_h, new_w), dtype=dtype, order="C")
    #
    # First run
    #
    print(f"First pass {x_shape} (checking outputs)", sep="", end="")
    #
    print(".", sep="", end="")
    numpy_matrix_out = pad_numpy(vpadding, hpadding, matrix_in)
    print(".", sep="", end="")
    pad_cython(matrix_in, cython_matrix_out)
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
    numpy_pad_t = timeit(lambda: pad_numpy(vpadding, hpadding, matrix_in),
                         number=10) / 10
    print(".", sep="", end="")
    cython_pad_t = timeit(lambda: pad_cython(matrix_in, cython_matrix_out),
                          number=10) / 10
    print()
    min_t = np.min([numpy_pad_t, cython_pad_t])
    a = "*" if cython_pad_t == min_t else ""
    return [["numpy", "a", "cython"],
            ["{:6.4f}".format(numpy_pad_t),
             a,
             "{:6.4f}".format(cython_pad_t - numpy_pad_t),
             ]]


if __name__ == '__main__':
    # D(b, c, h, w, kn, kh, kw, vpadding, hpadding, vstride, hstride, vdilation, hdilation):
    layers = [
        # D(1, 1, 5, 5, 64, 3, 3, 1, 1, 2, 2, 1, 1),
        # AlexNet Cifar
        D(64, 3, 32, 32, 64, 3, 3, 1, 1, 2, 2, 1, 1),
        D(64, 64, 8, 8, 192, 3, 3, 1, 1, 1, 1, 1, 1),
        D(64, 192, 4, 4, 384, 3, 3, 1, 1, 1, 1, 1, 1),
        D(64, 384, 4, 4, 256, 3, 3, 1, 1, 1, 1, 1, 1),
        D(64, 256, 4, 4, 256, 3, 3, 1, 1, 1, 1, 1, 1),
        # AlexNet ImageNet
        D(64, 3, 227, 227, 96, 11, 11, 1, 1, 4, 4, 1, 1),
        D(64, 96, 27, 27, 256, 5, 5, 1, 1, 1, 1, 1, 1),
        D(64, 256, 13, 13, 384, 3, 3, 1, 1, 1, 1, 1, 1),
        D(64, 384, 13, 13, 384, 3, 3, 1, 1, 1, 1, 1, 1),
        D(64, 384, 13, 13, 256, 3, 3, 1, 1, 1, 1, 1, 1),
    ]
    backward_layers = []
    for layer in layers:
        # w <- y (kn * b * ho * wo)
        backward_layers.append(D(layer.c, layer.b, layer.h, layer.w, layer.kn, layer.ho, layer.wo,
                                 layer.vpadding, layer.hpadding, layer.vstride, layer.hstride,
                                 later.vdilation, layer.hdilation))
    layers += backward_layers
    t = None
    for layer in layers:
        _x_shape = (layer.b, layer.c, layer.h, layer.w)
        headers, values = time_pad(_x_shape, layer.vpadding, layer.hpadding)
        if t is None:
            t = Table(box=box.HORIZONTALS, show_header=True, header_style="blue")
            t.add_column("x shape")
            for h in headers:
                t.add_column(str(h), justify="right")
        row_list = [", ".join([str(x) for x in _x_shape]), ] + values
        t.add_row(*row_list)
    print("*************************************************")
    print("** {}  OMP_NUM_THREADS: {}".format(platform.node(), os.environ.get("OMP_NUM_THREADS", 1)))
    print("** All times, except numpy, compared to numpy.")
    c = Console()
    c.print(t)
