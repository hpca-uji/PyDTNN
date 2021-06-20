"""
Performance tests for transposing matrices

For running the tests run:
    python profile_conv2d_transpose_0231.py

"""

import os
import platform
from timeit import timeit

import numpy as np
from prettytable import PrettyTable

from pydtnn.cython_modules import transpose_0231_kji_cython, transpose_0231_ijk_cython


class D:
    # def __init__(self):
    #     """Default parameters"""
    #     self.b = 1  # Batch size
    #     self.c = 1  # Channels per layer
    #     self.h = 128  # Layers height
    #     self.w = 100  # Layers width
    #     self.kn = 1  # Number of filters
    #     self.kh = 16  # Filters weights height
    #     self.kw = 10  # Filters weights width
    #     self.vpadding = 1  # Vertical padding
    #     self.hpadding = 1  # Horizontal padding
    #     self.vstride = 1  # Vertical stride
    #     self.hstride = 1  # Horizontal stride
    #     self.vdilation = 1  # Vertical dilation
    #     self.hdilation = 1  # Horizontal dilation

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


def numpy_transpose_0231(original, transposed):
    transposed[...] = original.transpose((0, 2, 3, 1))


def time_transpose_0231(shape, dtype=np.float32):
    d0, d1, d2, d3 = shape
    original = np.random.rand(d0, d1, d2, d3).astype(dtype, order="C")
    numpy_transposed = np.empty((d0, d2, d3, d1), dtype, order="C")
    cython_transposed = np.empty((d0, d2, d3, d1), dtype, order="C")
    #
    # First run
    #
    print(f"First pass {shape} (checking outputs)", sep="", end="")
    #
    print(".", sep="", end="")
    numpy_transpose_0231(original, numpy_transposed)
    #
    print(".", sep="", end="")
    transpose_0231_ijk_cython(original, cython_transposed)
    assert np.allclose(numpy_transposed, cython_transposed), "numpy and cython outer differ"
    #
    print(".", sep="", end="")
    transpose_0231_kji_cython(original, cython_transposed)
    assert np.allclose(numpy_transposed, cython_transposed), "numpy and cython 2nd loop differ"
    print()
    #
    # Second run
    #
    print(f"Second pass {shape} (getting times)", sep="", end="")
    print(".", sep="", end="")
    numpy_transpose_0231_t = timeit(lambda: numpy_transpose_0231(original, numpy_transposed),
                                    number=10) / 10
    print(".", sep="", end="")
    cython_transposed_0231_kji_t = timeit(lambda: transpose_0231_kji_cython(original, numpy_transposed),
                                          number=10) / 10
    print(".", sep="", end="")
    cython_transposed_0231_ijk_t = timeit(lambda: transpose_0231_ijk_cython(original, numpy_transposed),
                                          number=10) / 10
    print()
    min_t = np.min([numpy_transpose_0231_t, cython_transposed_0231_ijk_t, cython_transposed_0231_kji_t])
    a = "*" if cython_transposed_0231_kji_t == min_t else ""
    b = "*" if cython_transposed_0231_ijk_t == min_t else ""
    return [["numpy", "a", "cython kji", "c", "cython ijk"],
            ["{:6.4f}".format(numpy_transpose_0231_t),
             a,
             "{:6.4f}".format(cython_transposed_0231_kji_t - numpy_transpose_0231_t),
             b,
             "{:6.4f}".format(cython_transposed_0231_ijk_t - numpy_transpose_0231_t),
             ]]


if __name__ == '__main__':
    # D(b, c, h, w, kn, kh, kw, vpadding, hpadding, vstride, hstride, vdilation, hdilation)
    layersiAlexnet = [
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
    b = 32
    layers = [
        D(b, 3, 224, 224, 64, 7, 7, 3, 3, 2, 2, 1, 1),
        D(b, 64, 56, 56, 64, 1, 1, 0, 0, 1, 1, 1, 1),
        D(b, 64, 56, 56, 64, 3, 3, 1, 1, 1, 1, 1, 1),
        D(b, 64, 56, 56, 256, 1, 1, 0, 0, 1, 1, 1, 1),
        D(b, 64, 56, 56, 256, 1, 1, 0, 0, 1, 1, 1, 1),
        D(b, 256, 56, 56, 64, 1, 1, 0, 0, 1, 1, 1, 1),
        D(b, 64, 56, 56, 64, 3, 3, 1, 1, 1, 1, 1, 1),
        D(b, 64, 56, 56, 256, 1, 1, 0, 0, 1, 1, 1, 1),
        D(b, 256, 56, 56, 64, 1, 1, 0, 0, 1, 1, 1, 1),
        D(b, 64, 56, 56, 64, 3, 3, 1, 1, 1, 1, 1, 1),
        D(b, 64, 56, 56, 256, 1, 1, 0, 0, 1, 1, 1, 1),
        D(b, 256, 56, 56, 128, 1, 1, 0, 0, 1, 1, 1, 1),
        D(b, 128, 56, 56, 128, 3, 3, 1, 1, 2, 2, 1, 1),
        D(b, 128, 28, 28, 512, 1, 1, 0, 0, 1, 1, 1, 1),
        D(b, 256, 56, 56, 512, 1, 1, 0, 0, 2, 2, 1, 1),
        D(b, 512, 28, 28, 128, 1, 1, 0, 0, 1, 1, 1, 1),
        D(b, 128, 28, 28, 128, 3, 3, 1, 1, 1, 1, 1, 1),
        D(b, 128, 28, 28, 512, 1, 1, 0, 0, 1, 1, 1, 1),
        D(b, 512, 28, 28, 128, 1, 1, 0, 0, 1, 1, 1, 1),
        D(b, 128, 28, 28, 128, 3, 3, 1, 1, 1, 1, 1, 1),
        D(b, 128, 28, 28, 512, 1, 1, 0, 0, 1, 1, 1, 1),
        D(b, 512, 28, 28, 128, 1, 1, 0, 0, 1, 1, 1, 1),
        D(b, 128, 28, 28, 128, 3, 3, 1, 1, 1, 1, 1, 1),
        D(b, 128, 28, 28, 512, 1, 1, 0, 0, 1, 1, 1, 1),
        D(b, 512, 28, 28, 256, 1, 1, 0, 0, 1, 1, 1, 1),
        D(b, 256, 28, 28, 256, 3, 3, 1, 1, 2, 2, 1, 1),
        D(b, 256, 14, 14, 1024, 1, 1, 0, 0, 1, 1, 1, 1),
        D(b, 512, 28, 28, 1024, 1, 1, 0, 0, 2, 2, 1, 1),
        D(b, 1024, 14, 14, 256, 1, 1, 0, 0, 1, 1, 1, 1),
        D(b, 256, 14, 14, 256, 3, 3, 1, 1, 1, 1, 1, 1),
        D(b, 256, 14, 14, 1024, 1, 1, 0, 0, 1, 1, 1, 1),
        D(b, 1024, 14, 14, 256, 1, 1, 0, 0, 1, 1, 1, 1),
        D(b, 256, 14, 14, 256, 3, 3, 1, 1, 1, 1, 1, 1),
        D(b, 256, 14, 14, 1024, 1, 1, 0, 0, 1, 1, 1, 1),
        D(b, 1024, 14, 14, 256, 1, 1, 0, 0, 1, 1, 1, 1),
        D(b, 256, 14, 14, 256, 3, 3, 1, 1, 1, 1, 1, 1),
        D(b, 256, 14, 14, 1024, 1, 1, 0, 0, 1, 1, 1, 1),
        D(b, 1024, 14, 14, 256, 1, 1, 0, 0, 1, 1, 1, 1),
        D(b, 256, 14, 14, 256, 3, 3, 1, 1, 1, 1, 1, 1),
        D(b, 256, 14, 14, 1024, 1, 1, 0, 0, 1, 1, 1, 1),
        D(b, 1024, 14, 14, 256, 1, 1, 0, 0, 1, 1, 1, 1),
        D(b, 256, 14, 14, 256, 3, 3, 1, 1, 1, 1, 1, 1),
        D(b, 256, 14, 14, 1024, 1, 1, 0, 0, 1, 1, 1, 1),
        D(b, 1024, 14, 14, 512, 1, 1, 0, 0, 1, 1, 1, 1),
        D(b, 512, 14, 14, 512, 3, 3, 1, 1, 2, 2, 1, 1),
        D(b, 512, 7, 7, 2048, 1, 1, 0, 0, 1, 1, 1, 1),
        D(b, 1024, 14, 14, 2048, 1, 1, 0, 0, 2, 2, 1, 1),
        D(b, 2048, 7, 7, 512, 1, 1, 0, 0, 1, 1, 1, 1),
        D(b, 512, 7, 7, 512, 3, 3, 1, 1, 1, 1, 1, 1),
        D(b, 512, 7, 7, 2048, 1, 1, 0, 0, 1, 1, 1, 1),
        D(b, 2048, 7, 7, 512, 1, 1, 0, 0, 1, 1, 1, 1),
        D(b, 512, 7, 7, 512, 3, 3, 1, 1, 1, 1, 1, 1),
        D(b, 512, 7, 7, 2048, 1, 1, 0, 0, 1, 1, 1, 1),
    ]
    backward_layers = []
    for layer in layers:
        # if layer.vstride != 1 or layer.hstride != 1:
        #     continue
        # w <- y (kn * b * ho * wo)
        backward_layers.append(D(layer.c, layer.b, layer.h, layer.w, layer.kn, layer.ho, layer.wo,
                                 layer.vpadding, layer.hpadding, layer.vstride, layer.hstride,
                                 layer.vdilation, layer.hdilation))
    layers += backward_layers
    t = None
    for layer in layers:
        shape = (layer.b, layer.kn, layer.ho, layer.wo)
        headers, values = time_transpose_0231(shape)
        if t is None:
            t = PrettyTable(['shape', ] + headers)
            # t.set_style(PLAIN_COLUMNS)
            t.align = "r"
        t.add_row([", ".join([str(x) for x in shape]), ] + values)
    print("*************************************************")
    print("** {}  OMP_NUM_THREADS: {}".format(platform.node(), os.environ.get("OMP_NUM_THREADS", 1)))
    print("** All times, except numpy, compared to numpy.")
    print(t)
