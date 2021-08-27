"""
PyDTNN convWinograd module
"""

#
#  This file is part of Python Distributed Training of Neural Networks (PyDTNN)
#
#  Copyright (C) 2021 Universitat Jaume I
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
#  with this program. If not, see <https://www.gnu.org/licenses/>.
#

import ctypes
import math
import platform
import weakref
from contextlib import suppress

import numpy as np

# from pydtnn.tracers import PYDTNN_OPS_EVENT, PYDTNN_OPS_EVENTS, ...
from pydtnn.utils import load_library
from pydtnn.utils.best_pad import best_pad
from pydtnn.utils.memory_cache import MemoryCache
from pydtnn.utils.best_of import BestOf


try:
   load_library("convwinograd")
   is_conv_winograd_available = True
except ImportError:
   is_conv_winograd_available = False


class ConvWinograd:
    """
    Exposes the libconvWinograd functions following the PyDTNN conventions.

    Methods
    -------
    winograd(weights, x, biases, vpadding, hpadding, 
             vstride, hstride, vdilation, hdilation)
        calls the appropriate winograd function from libconvWinograd.so to perform
        the Toom-Cook based convolution.

    Examples
    --------
    See __usage_example__() method for an example of use. This example can be
    run with: 'python conv_winograd.py'

    Tests
    -----
    To perform the tests, run the following command from the current directory:
        python -m unittest tests.convWinogradTestcase

    (see tests/winograd.py for more instructions on testing)
    """

    lib_cw = None  # will link to the libconvWinograd.so library

    def __init__(self, kh, kw, vstride, hstride, vdilation, hdilation, winograd_tile_size=2,
                 dtype=np.float32, debug=False, parent_layer=None):
        """
        Loads the libconvWinograd.so library.

        Parameters
        ----------
        kh : kernel height

        kw : kernel width

        vstride : vertical stride

        hstride : horizontal stride

        vdilation : vertical dilation

        hdilation : horizontal dilation

        dtype : data type
            The element data type being used on all the matrices.
        debug : boolean
            Whether to print debug information or not.
        parent_layer: layer
            The layer that is using it (for tracing purposes).
        """

        def get_winograd_function(m, r):
            # choose the appropriate convWinograd function depending on the architecture and the data type being used
            if platform.machine() == 'aarch64':
                if self.dtype == np.float32:
                    try:
                        x_winograd_nchw = getattr(lib_cw, f"sconv_winograd{m}x{m}_{r}x{r}_nchw")
                        return (self._conv_winograd_c_nchw, x_winograd_nchw)
                    except AttributeError:
                        print(f"Winograd sconv_winograd{m}x{m}_{r}x{r}_nchw routine not found. Fallback to numpy version!")
                        return (self._conv_winograd_numpy_nchw, None)
                else:
                    raise NotImplementedError(f"Type {str(self.dtype)} not supported by this version of libconvWinograd!")
            elif platform.machine() == 'x86_64':
                if self.dtype == np.float32:
                    try:
                        x_winograd_nchw = getattr(self.lib_cw, f"sconv_winograd{m}x{m}_{r}x{r}_nchw")
                        return (self._conv_winograd_c_nchw, x_winograd_nchw)
                    except AttributeError:
                        print(f"Winograd sconv_winograd{m}x{m}_{r}x{r}_nchw routine not found. Fallback to numpy version!")
                        return (self._conv_winograd_numpy_nchw, None)
                else:
                    raise NotImplementedError(f"Type {str(self.dtype)} not supported by this version of libconvWinograd!")
            else:
                raise NotImplementedError(f"Platform '{str(platform.machine())}' not yet supported")

        # Parent layer
        if parent_layer is not None:
            self.get_parent_layer = weakref.ref(parent_layer)
            enable_best_of = self.get_parent_layer().model.enable_best_of
        else:
            enable_best_of = False

        if isinstance(dtype, type):
            self.dtype = dtype
        else:
            try:
                self.dtype = {'float32': np.float32, 'float64': np.float64}[dtype]
            except KeyError:
                raise NotImplementedError("dtype '{}' not recognized".format(dtype))

        if ConvWinograd.lib_cw is None:
            ConvWinograd.lib_cw = load_library("convwinograd")

        if (kh, kw) == (2, 2) and (vstride, hstride) == (1, 1) and (vdilation, hdilation) == (1, 1):
            # F(3x3, 2x2)
            self.m, self.r = 3, 2   # Winograd output tile (m x m) and filter (r x r) sizes
            self.g_3x3_2x2  = np.array([[      1,      0 ],
                                        [  1./2.,  1./2. ],
                                        [  1./2., -1./2. ],
                                        [      0,      1 ]],
                                         dtype=self.dtype, order="C")  # G
            self.bt_3x3_2x2 = np.array([[      1,      0,     -1,      0 ],
                                        [      0,      1,      1,      0 ],
                                        [      0,     -1,      1,      0 ],
                                        [      0,     -1,      0,      1 ]],
                                         dtype=self.dtype, order="C")  # Transpose of B
            self.at_3x3_2x2 = np.array([[      1,      1,      1,      0 ],
                                        [      0,      1,     -1,      0 ],
                                        [      0,      1,      1,      1 ]],
                                         dtype=self.dtype, order="C")  # Transpose of A
            self._conv_winograd_3x3_2x2_nchw_func, self._conv_winograd_3x3_2x2_nchw_func_int = \
                                                                               get_winograd_function(3, 2)

        elif (kh, kw) == (3, 3) and (vstride, hstride) == (1, 1) and (vdilation, hdilation) == (1, 1) and \
             (winograd_tile_size == 2 or enable_best_of):
            # F(2x2, 3x3)
            self.m, self.r = 2, 3   # Winograd output tile (m x m) and filter (r x r) sizes
            self.g_2x2_3x3  = np.array([[      1,      0,      0 ],
                                        [  1./2.,  1./2.,  1./2. ],
                                        [  1./2., -1./2.,  1./2. ],
                                        [      0,      0,      1 ]],
                                         dtype=self.dtype, order="C")  # G
            self.bt_2x2_3x3 = np.array([[      1,      0,     -1,      0 ],
                                        [      0,      1,      1,      0 ],
                                        [      0,     -1,      1,      0 ],
                                        [      0,      1,      0,     -1 ]],
                                         dtype=self.dtype, order="C")  # Transpose of B
            self.at_2x2_3x3 = np.array([[      1,      1,      1,      0 ],
                                        [      0,      1,     -1,     -1 ]],
                                         dtype=self.dtype, order="C")  # Transpose of A
            self._conv_winograd_2x2_3x3_nchw_func, self._conv_winograd_2x2_3x3_nchw_func_int = \
                                                                               get_winograd_function(2, 3)

        else:
            raise NotImplementedError(f"Winograd not implemented for kernel {kh}x{kw}")

        if (kh, kw) == (3, 3) and (vstride, hstride) == (1, 1) and (vdilation, hdilation) == (1, 1) and \
           (winograd_tile_size == 4 or enable_best_of):
            # F(4x4, 3x3)
            self.m, self.r = 4, 3   # Winograd output tile (m x m) and filter (r x r) sizes
            self.g_4x4_3x3  = np.array([[  1./4.,      0,      0 ],
                                        [ -1./6., -1./6., -1./6. ],
                                        [ -1./6.,  1./6., -1./6. ],
                                        [ 1./24., 1./12.,  1./6. ],
                                        [ 1./24.,-1./12.,  1./6. ],
                                        [      0,      0,      1 ]],
                                         dtype=self.dtype, order="C")  # G
            self.bt_4x4_3x3 = np.array([[      4,      0,     -5,      0,      1,      0 ],
                                        [      0,     -4,     -4,      1,      1,      0 ],
                                        [      0,      4,     -4,     -1,      1,      0 ],
                                        [      0,     -2,     -1,      2,      1,      0 ],
                                        [      0,      2,     -1,     -2,      1,      0 ],
                                        [      0,      4,      0,     -5,      0,      1 ]],
                                         dtype=self.dtype, order="C")  # Transpose of B
            self.at_4x4_3x3 = np.array([[      1,      1,      1,      1,      1,      0 ],
                                        [      0,      1,     -1,      2,     -2,      0 ],
                                        [      0,      1,      1,      4,      4,      0 ],
                                        [      0,      1,     -1,      8,     -8,      1 ]],
                                         dtype=self.dtype, order="C")  # Transpose of A
            self._conv_winograd_4x4_3x3_nchw_func, self._conv_winograd_4x4_3x3_nchw_func_int = \
                                                                               get_winograd_function(4, 3)

        # The x_padded and output matrices are also cached, but only on the current instance
        # self.x_padded_cache = MemoryCache(lambda shape: np.zeros(shape, self.dtype, order="C"))
        self.y_cache = MemoryCache(lambda shape: np.zeros(shape, self.dtype, order="C"))
        self.u_cache = MemoryCache(lambda shape: np.zeros(shape, self.dtype, order="C"))
        self.v_cache = MemoryCache(lambda shape: np.zeros(shape, self.dtype, order="C"))
        self.m_cache = MemoryCache(lambda shape: np.zeros(shape, self.dtype, order="C"))
        self.d_cache = MemoryCache(lambda shape: np.zeros(shape, self.dtype, order="C"))

        # Debug
        self.debug = debug

        if enable_best_of and self.r == 3:
            self.conv_winograd_nchw = BestOf(
                name="Winograd functions",
                alternatives=[
                    ("winograd_2x2_3x3", self._conv_winograd_2x2_3x3_nchw),
                    ("winograd_4x4_3x3", self._conv_winograd_4x4_3x3_nchw),
                ],
                get_problem_size=lambda *args, **kwargs: tuple(list(args[0].shape) + list(args[1].shape)),
            )
        else:
            self.conv_winograd_nchw = getattr(self, f"_conv_winograd_{self.m}x{self.m}_{self.r}x{self.r}_nchw")

    def _conv_winograd_3x3_2x2_nchw(self, *args, **kwargs):
       return self._conv_winograd_nchw_gen(3, 2, *args, **kwargs)

    def _conv_winograd_2x2_3x3_nchw(self, *args, **kwargs):
       return self._conv_winograd_nchw_gen(2, 3, *args, **kwargs)

    def _conv_winograd_4x4_3x3_nchw(self, *args, **kwargs):
       return self._conv_winograd_nchw_gen(4, 3, *args, **kwargs)

    def _conv_winograd_nchw_gen(self, m, r, *args, **kwargs):
       return getattr(self, f"_conv_winograd_{m}x{m}_{r}x{r}_nchw_func")\
                                           (m, r, getattr(self, f"g_{m}x{m}_{r}x{r}"),
                                                  getattr(self, f"bt_{m}x{m}_{r}x{r}"),
                                                  getattr(self, f"at_{m}x{m}_{r}x{r}"),
                                                  getattr(self, f"_conv_winograd_{m}x{m}_{r}x{r}_nchw_func_int"),
                                                  *args, **kwargs)

    def _conv_winograd_numpy_nchw(self, m, r, g, bt, at, x_winograd_nchw,
                                  weights, x, biases=None, vpadding=0, hpadding=0,
                                  vstride=1, hstride=1, vdilation=1, hdilation=1,
                                  relu=False, bn=False, running_mean=None, inv_std=None,
                                  gamma=None, beta=None):
        n,  ci, hi, wi = x.shape
        co, ci, kh, kw = weights.shape

        s = r - 1      # Winograd sliding window stride
        t = m + r - 1  # Winograd sliding window size t x t

        if (kh, kw) != (r, r):
            raise ValueError(f"Kernel size {kh}x{kw} supported by this version of Winograd, kernel size should be ({r}x{r})!")

        if (vstride, hstride) != (1, 1):
            raise ValueError(f"Stride {vstride}x{hstride} supported by this version of Winograd, stride should be (1,1)!")

        if (vdilation, hdilation) != (1, 1):
            raise ValueError(f"Dilation {vdilation}x{hdilation} supported by this version of Winograd, dilation should be (1,1)!")

        ho = (hi + 2 * vpadding - vdilation * (kh - 1) - 1) // vstride + 1
        wo = (wi + 2 * hpadding - hdilation * (kw - 1) - 1) // hstride + 1

        tile_h = (hi + 2 * vpadding - kh) // m + 1
        tile_w = (wi + 2 * hpadding - kw) // m + 1

        y = self.y_cache[(n, co, ho, wo)]    # Output
        u = self.u_cache[(t, t, co, ci)]     # Workspace for G * g * G^T
        v = self.v_cache[(t, t, ci, (n * tile_h * tile_w))]
        # m_= self.m_cache[(t, t, co, (n * tile_h * tile_w))]
        d = self.d_cache[(t, t)]

        for k in range(co):
            for c in range(ci):
                # U = G  * g * G^T
                u[..., k, c] = (g @ weights[k, c, ...]) @ g.T

        # 1.1) First alternative: padding first
        # x_padded = best_pad(x, vpadding, hpadding)
        # _, _, hi, wi = x_padded.shape
        #
        # for c in range(ci):
        #     for b in range(n):
        #         for h in range(tile_h):
        #             for w in range(tile_w):
        #                 hh, ww = h * s, w * s
        #                 th, tw = min(hi-hh,t), min(wi-ww,t)
        #                 d[:th, :tw] = x_padded[b, c, hh:hh+th, ww:ww+tw]
        #                 v[..., c, b * tile_h * tile_w + h * tile_w + w] = (self.bt @ d) @ self.bt.T

        # 1.2) Second alternative: avoid padding
        for c in range(ci):
            for b in range(n):
                for h in range(tile_h):
                    hh_= min(hi, h * s - vpadding)
                    hh, fh = (hh_, 0) if hh_ > 0 else (0, min(-hh_, t))
                    oh = max(min(t, hi - hh) - min(t, fh), 0)

                    for w in range(tile_w):
                        ww_= min(wi, w * s - hpadding)
                        ww, fw = (ww_, 0) if ww_ > 0 else (0, min(-ww_, t))
                        ow = max(min(t, wi - ww) - min(t, fw), 0)

                        if hh < hh+oh and ww < ww+ow:
                            d[fh:fh+oh, fw:fw+ow] = x[b, c, hh:hh+oh, ww:ww+ow]

                        #   0  0  0
                        #   X  X  X
                        #   X  X  X
                        if 0 <= fh:
                            d[:fh, ...] = 0

                        #   0  0  0
                        #   X  X  X
                        #   0  0  0
                        if fh+oh < t:
                            d[fh+oh:, ...] = 0

                        #   0  0  0
                        #   0  X  X
                        #   0  0  0
                        if 0 <= fw:
                            d[fh:fh+oh, :fw] = 0

                        #   0  0  0
                        #   0  X  0
                        #   0  0  0
                        if fw+ow < t:
                            d[fh:fh+oh, fw+ow:] = 0

                        v[..., c, b * tile_h * tile_w + h * tile_w + w] = (bt @ d) @ bt.T

        # 2.1) Firt alternative: np.einsum
        m_= np.einsum('... i j, ... j k -> ... i k', u, v)

        # 2.2) Second alternative: matmul
        # for i in range(t):
        #     for j in range(t):
        #         m_[i, j] = u[i, j] @ v[i, j]
        
        for k in range(co):
            for b in range(n):
                for h in range(tile_h):
                    for w in range(tile_w): 
                        z = (at @ m_[..., k, b * tile_h * tile_w + h * tile_w + w]) @ at.T
                        hh, ww = h * s, w * s
                        y[b, k, hh:hh+m, ww:ww+m] = z[:min(m, ho-hh), :min(m, wo-ww)]

            if biases is not None:
                y[:, k, ...] += biases[k]

            if bn:
                y[:, k, ...] = (((y[:, k, ...]  - running_mean[k]) * inv_std[k]) * gamma[k]) + beta[k];

        if relu:
           y[y < 0] = 0

        return y

    def _conv_winograd_c_nchw(self, m, r, g, bt, at, x_winograd_nchw,
                              weights, x, biases=None, vpadding=0, hpadding=0,
                              vstride=1, hstride=1, vdilation=1, hdilation=1,
                              relu=False, bn=False, running_mean=None, inv_std=None,
                              gamma=None, beta=None):
        n,  ci, hi, wi = x.shape
        co, ci, kh, kw = weights.shape

        s = r - 1      # Winograd sliding window stride
        t = m + r - 1  # Winograd sliding window size t x t

        if (kh, kw) != (r, r):
            raise ValueError(f"Kernel size {kh}x{kw} supported by this version of Winograd, kernel size should be ({r}x{r})!")

        if (vstride, hstride) != (1, 1):
            raise ValueError(f"Stride {vstride}x{hstride} supported by this version of Winograd, stride should be (1,1)!")

        if (vdilation, hdilation) != (1, 1):
            raise ValueError(f"Dilation {vdilation}x{hdilation} supported by this version of Winograd, dilation should be (1,1)!")

        ho = (hi + 2 * vpadding - vdilation * (kh - 1) - 1) // vstride + 1
        wo = (wi + 2 * hpadding - hdilation * (kw - 1) - 1) // hstride + 1

        tile_h = (hi + 2 * vpadding - kh) // m + 1
        tile_w = (wi + 2 * hpadding - kw) // m + 1

        y = self.y_cache[(n, co, ho, wo)]    # Output
        u = self.u_cache[(t, t, co, ci)]     # Workspace for G * g * G^T
        v = self.v_cache[(t, t, ci, (n * tile_h * tile_w))]
        m1= self.m_cache[(t, t, co, (n * tile_h * tile_w))]
        m2= self.m_cache[(co, (n * tile_h * tile_w))]
        d = self.d_cache[(t, t)]

        ldD1, ldD2, ldD3 = ci * hi * wi, hi * wi, wi
        ldF1, ldF2, ldF3 = ci * kh * kw, kh * kw, kw
        ldY1, ldY2, ldY3 = co * ho * wo, ho * wo, wo

        x_winograd_nchw(ctypes.c_uint(n), ctypes.c_uint(co), ctypes.c_uint(ci),
                        ctypes.c_uint(hi), ctypes.c_uint(wi),
                        ctypes.c_uint(kh), ctypes.c_uint(kw),
                        ctypes.c_uint(vpadding), ctypes.c_uint(hpadding),
                        ctypes.c_void_p(x.ctypes.data), ctypes.c_uint(ldD1), ctypes.c_uint(ldD2), ctypes.c_uint(ldD3),
                        ctypes.c_void_p(weights.ctypes.data),ctypes.c_uint(ldF1), ctypes.c_uint(ldF2), ctypes.c_uint(ldF3),
                        ctypes.c_void_p(y.ctypes.data), ctypes.c_uint(ldY1), ctypes.c_uint(ldY2), ctypes.c_uint(ldY3),
                        ctypes.c_void_p(None if biases is None else biases.ctypes.data),
                        ctypes.c_void_p(bt.ctypes.data), ctypes.c_void_p(g.ctypes.data), ctypes.c_void_p(at.ctypes.data),
                        ctypes.c_void_p(u.ctypes.data), ctypes.c_void_p(v.ctypes.data),
                        ctypes.c_void_p(m1.ctypes.data), ctypes.c_void_p(m2.ctypes.data),
                        ctypes.c_char((b'F', b'T')[relu]), ctypes.c_char((b'F', b'T')[bn]),
                        ctypes.c_void_p(None if running_mean is None else running_mean.ctypes.data),
                        ctypes.c_void_p(None if inv_std is None else inv_std.ctypes.data),
                        ctypes.c_void_p(None if gamma is None else gamma.ctypes.data),
                        ctypes.c_void_p(None if beta is None else beta.ctypes.data))
        return y


def __usage_example__():
    # Imports for this usage example (not required otherwise)
    from timeit import timeit
    from pydtnn.cython_modules import im2col_nchw_cython
    # Default parameters (1st layer AlexNet for Cifar10)
    b = 64  # Batch size
    c = 3  # Channels per layer
    h = 32  # Layers height
    w = 32  # Layers width
    kn = 64  # Number of filters
    kh = 3  # Filters weights height
    kw = 3  # Filters weights width
    vpadding = 3  # Vertical padding
    hpadding = 3  # Horizontal padding
    vstride = 1  # Vertical stride
    hstride = 1  # Horizontal stride
    vdilation = 1  # Vertical dilation
    hdilation = 1  # Horizontal dilation
    # Create weights, x, and biases matrices from previous parameters. If no biases
    # matrix is provided, a proper one filled with zeros will be automatically
    # created.
    weights = np.zeros((kn, c, kh, kw)).astype(np.float32, order='C')
    weights[0][0][0][0] = -100.89
    weights[1][1][1][1] = -322.0
    weights[2][2][2][2] = -334.0
    x = np.ones((b, c, h, w)).astype(np.float32, order='C')
    ho = (h + 2 * vpadding - vdilation * (kh - 1) - 1) // vstride + 1
    wo = (w + 2 * hpadding - hdilation * (kw - 1) - 1) // hstride + 1
    biases = (np.ones((kn, b * ho * wo)) * 10).astype(np.float32, order='C')
    biases_wg = (np.ones((kn)) * 10).astype(np.float32, order='C')
    print("Using conv_winograd to compute weights * x + biases...")
    r = False
    conv_winograd = ConvWinograd(kh, kw, vstride, hstride, vdilation, hdilation, debug=False)
    conv_winograd_result = conv_winograd.conv_winograd_nchw(weights, x, biases_wg,
                                         vpadding=vpadding, hpadding=hpadding,
                                         vstride=vstride, hstride=hstride,
                                         vdilation=vdilation, hdilation=hdilation, relu=r)
    # print(conv_winograd_result)
    print("Sum: ", conv_winograd_result.sum())
    conv_winograd_t = timeit(lambda: conv_winograd.conv_winograd_nchw(weights, x, biases_wg,
                                        vpadding=vpadding, hpadding=hpadding,
                                        vstride=vstride, hstride=hstride,
                                        vdilation=vdilation, hdilation=hdilation, relu=r),
                                        number=10) / 10
    print("conv_winograd time: {:.4f}".format(conv_winograd_t))
    print()
    print("Using im2col and mm...")
    x_c = im2col_nchw_cython(x, kh, kw, vpadding, hpadding, vstride, hstride, vdilation, hdilation)
    w_c = weights.reshape(kn, -1)
    im2col_mm_result = w_c @ x_c + biases
    im2col_mm_result = im2col_mm_result.reshape(kn, -1, ho, wo).transpose(1, 0, 2, 3)
    if r:
        im2col_mm_result = np.maximum(im2col_mm_result, 0)
    # print(im2col_mm_result)
    print("Sum: ", im2col_mm_result.sum())
    print("np.allclose: ", np.allclose(conv_winograd_result, im2col_mm_result))
    # im2col_t = timeit(lambda: im2col_nchw_cython(x, kh, kw, vpadding, hpadding, vstride, hstride,
    #                                              vdilation, hdilation), number=10) / 10
    # print("im2col time: {:.4f}".format(im2col_t))
    mm_t = timeit(lambda: w_c @ im2col_nchw_cython(x, kh, kw, vpadding, hpadding, vstride, hstride, vdilation, hdilation) \
                       + biases, number=10) / 10
    print("mm time: {:.4f}".format(mm_t))

    # """
    n = 64
    c = k = 64
    h = w = 32
    vpadd = hpadd = 6
    for nn in range(16, n, 16):
        for cc in range(16, c, 16):
            for kk in range(16, k, 16):
                for hh in range(8, h, 4):
                    for ww in range(8, w, 4):
                        for vpadding in range(1, vpadd):
                            for hpadding in range(1, hpadd):
                                print(nn, cc, kk, hh, ww, vpadding, hpadding, end="")
                                weights = np.random.rand(kk, cc, kh, kw).astype(np.float32, order='C')
                                x = np.random.rand(nn, cc, hh, ww).astype(np.float32, order='C')
                                ho = (hh + 2 * vpadding - vdilation * (kh - 1) - 1) // vstride + 1
                                wo = (ww + 2 * hpadding - hdilation * (kw - 1) - 1) // hstride + 1
                                biases = (np.ones((kk, nn * ho * wo)) * 10).astype(np.float32, order='C')
                                biases_wg = (np.ones((kk)) * 10).astype(np.float32, order='C')
                                conv_winograd = ConvWinograd(kh, kw, vstride, hstride, vdilation, hdilation, debug=False)
                                conv_winograd_t = timeit(lambda: conv_winograd.conv_winograd_nchw(weights, x, biases_wg,
                                                                    vpadding=vpadding, hpadding=hpadding,
                                                                    vstride=vstride, hstride=hstride,
                                                                    vdilation=vdilation, hdilation=hdilation),
							    number=10) / 10
                                print(" conv_winograd time: {:.4f} ".format(conv_winograd_t), end="")
                                w_c = weights.reshape(kk, -1)
                                im2col_t = timeit(lambda: w_c @ im2col_nchw_cython(x, kh, kw, vpadding, hpadding, vstride, hstride, vdilation, hdilation) \
                                                   + biases, number=10) / 10
                                print("mm time: {:.4f}".format(im2col_t), end="")
                                print((" WINOGR", " IM2COL")[conv_winograd_t > im2col_t], im2col_t/conv_winograd_t)
    # """

if __name__ == "__main__":
    __usage_example__()
