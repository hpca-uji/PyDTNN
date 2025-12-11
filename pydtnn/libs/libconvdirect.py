"""
PyDTNN convDirect module
"""

import ctypes
import sys
import weakref

import numpy as np

from pydtnn.utils.tensor import TensorFormat, encode_shape, decode_shape
from pydtnn.utils import load_library
from pydtnn.backends.cpu.utils.im2row_nhwc_cython import im2row_nhwc_cython

try:
    load_library("convDirect")
    is_conv_direct_available = True
except Exception:
    is_conv_direct_available = False


class ConvDirect:
    """
    Exposes the libconvDirect functions following the PyDTNN conventions.

    Methods
    -------
    conv_direct(weights, x, out, vpadding, hpadding,
                vstride, hstride, vdilation, hdilation)
        calls the appropriate convDirect function from libconvDirect.so to perform
        the selected convDirect method.

    Examples
    --------
    See __usage_example__() method for an example of use. This example can be
    run with: 'python conv_direct.py'

    Tests
    -----
    To perform the tests, run the following command from the current directory:
        python -m unittest tests.convWinogradTestcase

    (see tests/winograd.py for more instructions on testing)
    """

    lib_cd = None  # will link to the libconvDirect.so library

    def _set_methods(self, method_name):
        return

    def __init__(self, method_name, dtype: np.dtype = np.dtype(np.float32), tensor_format=TensorFormat.NHWC,
                 debug=False, parent_layer=None):
        """
        Loads the libconvDirect.so library.

        Parameters
        ----------
        method : string
            Name of the method to be used
        dtype : data type
            The element data type being used on all the matrices.
        tensor_format : enumerate
            The tensor format to be used
        debug : boolean
            Whether to print debug information or not.
        parent_layer: layer
            The layer that is using it.
        """

        # self properties from parameters
        self.dtype = dtype
        self.tensor_format = tensor_format
        self.debug = debug

        # Parent layer
        if parent_layer is not None:
            self.get_parent_layer = weakref.ref(parent_layer)
            self.evaluate_only = self.get_parent_layer().model.evaluate_only  # type: ignore
            if not self.evaluate_only:
                raise NotImplementedError("The convDirect module only works in evaluate_only mode!")
        else:
            self.evaluate_only = True

        if ConvDirect.lib_cd is None:
            ConvDirect.lib_cd = load_library("convDirect")

        self._DT = ctypes.POINTER(ctypes.c_float)()
        self._FT = ctypes.POINTER(ctypes.c_float)()
        self._YT = ctypes.POINTER(ctypes.c_float)()

        try:
            [self._conv_direct_pre,
             self._conv_direct_kernel,
             self._conv_direct_post] = [getattr(self.__class__.lib_cd, method_name + suffix)
                                        for suffix in ('_pre', '_kernel', '_post')]
        except AttributeError as exc:
            raise NotImplementedError("Error: Method '{}' not supported by convDirect library.\n"
                                      "       Run convDirect_info to see the convDirect supported methods."
                                      "".format(method_name)) from exc

        self._reuse_processed_weights = False
        if self.evaluate_only and method_name.find("convdirect_block_blis") == 0:
            self._reuse_processed_weights = True
        self._weights_already_processed = False

    def encode_shape(self, shape):
        return encode_shape(shape, self.tensor_format)

    def decode_shape(self, shape):
        return decode_shape(shape, self.tensor_format)

    def conv_direct(self, weights: np.ndarray, x: np.ndarray,
                    # NOTE: "out" originally was called "biases"
                    out: np.ndarray | None = None,  # type: ignore
                    vpadding=0, hpadding=0,
                    vstride=1, hstride=1,
                    vdilation=1, hdilation=1,
                    relu=False, bn=False, running_mean=None,
                    inv_std=None, gamma=None, beta=None):

        n, ci, hi, wi = self.decode_shape(x.shape)

        if self.tensor_format is TensorFormat.NCHW:
            co, ci, kh, kw = weights.shape
        else:
            ci, kh, kw, co = weights.shape

        # TODO: Move to Conv2D_CPU (or remove)
        if out is None:
            ho = (hi + 2 * vpadding - vdilation * (kh - 1) - 1) // vstride + 1
            wo = (wi + 2 * hpadding - hdilation * (kw - 1) - 1) // hstride + 1
            bias_shape = self.encode_shape((n, co, ho, wo))
            out = np.zeros(bias_shape, weights.dtype, order="C")
        else:
            out = out[:n, :]

            match self.tensor_format:
                case TensorFormat.NCHW:
                    bb, knb, ho, wo = out.shape
                case TensorFormat.NHWC:
                    bb, ho, wo, knb = out.shape
                case tensor_format:
                    raise NotImplementedError(f"No support for {tensor_format} tensor format!")
            assert co == knb, "Number of filters must be the same!"
            assert n == bb, "Batch sizes must be the same!"
        out: np.ndarray

        # int t = n, Co = co, Ci = ci, Ho = h, Wo = w, Hf = r, Wf = s;
        (t, Co, Ci, Ho, Wo, Hf, Wf) = (n, co, ci, hi, wi, kh, kw)

        if self._reuse_processed_weights and self._weights_already_processed:
            self._DT = x.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
            self._YT = out.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        else:
            # #define CONVDIRECT_PRE_PARAMS \
            #     int t, int Co, int Ci,    \
            #     int Ho, int Wo,           \
            #     int Hf, int Wf,           \
            #     const DTYPE *D,           \
            #     const DTYPE *F,           \
            #     const DTYPE *Y,           \
            #     DTYPE **DT,               \
            #     DTYPE **FT,               \
            #     DTYPE **YT
            self._conv_direct_pre(ctypes.c_int(t), ctypes.c_int(Co), ctypes.c_int(Ci),
                                  ctypes.c_int(Ho), ctypes.c_int(Wo),
                                  ctypes.c_int(Hf), ctypes.c_int(Wf),
                                  ctypes.c_void_p(x.ctypes.data),
                                  ctypes.c_void_p(weights.ctypes.data),
                                  ctypes.c_void_p(out.ctypes.data),
                                  ctypes.byref(self._DT),
                                  ctypes.byref(self._FT),
                                  ctypes.byref(self._YT))
            self._weights_already_processed = True

        # #define CONVDIRECT_KERNEL_PARAMS  \
        #     int t, int Co, int Ci,        \
        #     int Ho, int Wo,               \
        #     int Hf, int Wf,               \
        #     int vpadding, int hpadding,   \
        #     int vstride, int hstride,     \
        #     int vdilation, int hdilation, \
        #     DTYPE alpha,                  \
        #     const DTYPE *DT,              \
        #     const DTYPE *FT,              \
        #     DTYPE beta,                   \
        #     DTYPE *YT
        self._conv_direct_kernel(ctypes.c_int(t), ctypes.c_int(Co), ctypes.c_int(Ci),
                                 ctypes.c_int(Ho), ctypes.c_int(Wo),
                                 ctypes.c_int(Hf), ctypes.c_int(Wf),
                                 ctypes.c_int(vpadding), ctypes.c_int(hpadding),
                                 ctypes.c_int(vstride), ctypes.c_int(hstride),
                                 ctypes.c_int(vdilation), ctypes.c_int(hdilation),
                                 ctypes.c_float(1.0),
                                 self._DT,
                                 self._FT,
                                 ctypes.c_float(1.0),
                                 self._YT)

        if not self._reuse_processed_weights:
            # #define CONVDIRECT_POST_PARAMS \
            #     int t, int Co, int Ci,     \
            #     int Ho, int Wo,            \
            #     int Hf, int Wf,            \
            #     DTYPE **DT,                \
            #     DTYPE **FT,                \
            #     DTYPE **YT,                \
            #     DTYPE *Y
            self._conv_direct_post(ctypes.c_int(t), ctypes.c_int(Co), ctypes.c_int(Ci),
                                   ctypes.c_int(Ho), ctypes.c_int(Wo),
                                   ctypes.c_int(Hf), ctypes.c_int(Wf),
                                   ctypes.c_float(1.0),
                                   ctypes.byref(self._DT),
                                   ctypes.byref(self._FT),
                                   ctypes.byref(self._YT),
                                   ctypes.c_void_p(out.ctypes.data))
        return out


def time_it_func(x: np.ndarray, w_c: np.ndarray, out: np.ndarray,
                 b: int, kn: int,
                 ho: int, wo: int, kh: int, kw: int,
                 vpadding: int, hpadding: int, vstride: int, hstride: int,
                 vdilation: int, hdilation: int,
                 ) -> int | float:
    res = np.zeros(((x.shape[0] * ho * wo), (x.shape[-1] * kh * kw)), dtype=x.dtype)
    im2row_nhwc_cython(x, res,
                       kh, kw, ho, wo,
                       vpadding, hpadding, vstride, hstride,
                       vdilation, hdilation)
    res = res @ w_c
    res += out.reshape(b * ho * wo, kn)
    return res


def __usage_example__():
    # Imports for this usage example (not required otherwise)
    from timeit import timeit
    from pydtnn.utils import random
    # Default parameters (1st layer AlexNet for Cifar10)
    b = 32  # Batch size
    c = 16  # Channels per layer
    h = 18  # Layers height
    w = 18  # Layers width
    kn = 16  # Number of filters
    kh = 3  # Filters weights height
    kw = 3  # Filters weights width
    vpadding = 1  # Vertical padding
    hpadding = 1  # Horizontal padding
    vstride = 1  # Vertical stride
    hstride = 1  # Horizontal stride
    vdilation = 1  # Vertical dilation
    hdilation = 1  # Horizontal dilation
    # Create weights, x, and out matrices from previous parameters. If no out
    # matrix is provided, a proper one filled with zeros will be automatically
    # created.
    random.seed(0)
    # weights[1][1][1][1] = -322.0
    # weights[2][2][2][2] = -334.0

    ho = (h + 2 * vpadding - vdilation * (kh - 1) - 1) // vstride + 1
    wo = (w + 2 * hpadding - hdilation * (kw - 1) - 1) // hstride + 1

    # NCHW --------------------------
    weights = random.random((c, kh, kw, kn)).astype(np.float32, order='C')
    x = random.random((b, h, w, c)).astype(np.float32, order='C')
    out = (np.ones((b, ho, wo, kn)) * 10).astype(np.float32, order='C')

    print("Using conv_direct to compute weights * x + out...")
    # conv_direct = ConvDirect("convdirect_im2row_nhwc_default")
    conv_direct = ConvDirect("convdirect_block_blis_nhwc_blis")
    # conv_direct = ConvDirect("convdirect_conv_gemm_nhwc_default")
    # from ipdb import launch_ipdb_on_exception
    # with launch_ipdb_on_exception():
    conv_direct_result = conv_direct.conv_direct(weights, x, out,
                                                 vpadding=vpadding, hpadding=hpadding,
                                                 vstride=vstride, hstride=hstride,
                                                 vdilation=vdilation, hdilation=hdilation)
    conv_direct_t = timeit(lambda: conv_direct.conv_direct(weights, x, out,
                                                           vpadding=vpadding, hpadding=hpadding,
                                                           vstride=vstride, hstride=hstride,
                                                           vdilation=vdilation, hdilation=hdilation),
                           number=10) / 10
    print("Using im2col and mm NHWC ...")

    x_c = np.zeros(((x.shape[0] * ho * wo), (x.shape[-1] * kh * kw)), dtype=x.dtype)
    im2row_nhwc_cython(x, x_c,
                       kh, kw, ho, wo,
                       vpadding, hpadding, vstride, hstride, vdilation, hdilation)
    w_c = weights.reshape(-1, kn)
    im2col_mm_result_nhwc = (x_c @ w_c + out.reshape(b * ho * wo, kn)).reshape(-1, ho, wo, kn)
    mm_t = timeit(
        lambda: time_it_func(x, w_c, out, b, kn, ho, wo, kh, kw, vpadding, hpadding, vstride, hstride, vdilation, hdilation),
        number=10) / 10

    print("conv_direct time: {:.4f}".format(conv_direct_t))
    print("im2row + mm time: {:.4f}".format(mm_t))
    print("Sum WINOGRAD NHWC: ", conv_direct_result.sum(), conv_direct_result.shape)
    print("Sum   IM2COL NHWC: ", im2col_mm_result_nhwc.sum(), im2col_mm_result_nhwc.shape)
    print("np.allclose NHWC: ", np.allclose(conv_direct_result, im2col_mm_result_nhwc, atol=1e-3))


if __name__ == "__main__":
    __usage_example__()
