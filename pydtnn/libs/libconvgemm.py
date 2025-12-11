"""
PyDTNN convGemm module
"""

import ctypes
import platform
import weakref

import numpy as np
from pydtnn.backends.cpu.utils.im2col_nchw_cython import im2col_nchw_cython

from pydtnn.utils import load_library

try:
    load_library("convGemm")
    is_conv_gemm_available = True
except Exception:
    is_conv_gemm_available = False


class ConvGemm:
    """
    Exposes the libconvGemm functions following the PyDTNN conventions.

    Methods
    -------
    conv_gemm(weights, x, out, vpadding, hpadding, vstride, hstride,
              vdilation, hdilation, biases)
        Calls the appropriate convGemm function from libconvGemm.so to perform a
        matrix matrix multiplication with an implicit im2col.

    Examples
    --------
    See __usage_example__() method for an example of use. This example can be
    run with: 'python conv_gemm.py'

    Tests
    -----
    To perform the tests, run the following command from the current directory:
        python -m unittest tests.ConvGemmTestCase

    (see tests/conv_gemm.py for more instructions on testing)
    """

    lib_cg = None  # will link to the libconvGemm.so library

    def __init__(self, dtype: np.dtype = np.dtype(np.float32), debug: bool = False, parent_layer=None):
        """
        Loads the libconvGemm.so library and creates the required auxiliary matrices ac_pack and bc_pack.

        Parameters
        ----------
        dtype : data type
            The element data type being used on all the matrices.
        debug : boolean
            Whether to print debug information or not.
        parent_layer: layer
            The layer that is using it (for tracing purposes).
        """
        self.dtype = dtype
        if ConvGemm.lib_cg is None:
            ConvGemm.lib_cg = load_library("convGemm")

        # Declare ac_pack and bc_pack and allocate space for them
        self.ac_pack = ctypes.POINTER(ctypes.c_float)()
        self.bc_pack = ctypes.POINTER(ctypes.c_float)()
        self.lib_cg.alloc_pack_buffs.restype = ctypes.c_int
        result = self.lib_cg.alloc_pack_buffs(ctypes.byref(self.ac_pack), ctypes.byref(self.bc_pack))
        if result == 1:
            raise MemoryError("Could not allocate space for ac_pack or bc_pack!")
        # Debug
        self.debug = debug
        # Parent layer
        if parent_layer is not None:
            self.get_parent_layer = weakref.ref(parent_layer)
        # Choose the appropriate convGemm function depending on the architecture and the data type being used
        if self.dtype == np.float32:
            self.x_conv_gemm_nhwc = self.lib_cg.sconvGemmNHWC
            self.x_deconv_gemm_nhwc = self.lib_cg.sconvGemmNHWC_back
            self.x_conv_gemm_nchw = self.lib_cg.sconvGemmNCHW
            self.x_deconv_gemm_nchw = self.lib_cg.sconvGemmNCHW_back
        else:
            raise TypeError(f"Type '{str(self.dtype)}' not supported by this version of libconvGemm!")

    def __del__(self):
        """Free the allocated matrices"""
        try:
            __free__(self.ac_pack)
            __free__(self.bc_pack)
        except AttributeError:
            pass

    def conv_gemm_nchw(self, weights: np.ndarray, x: np.ndarray,
                       # res originaly was called "biases"
                       out: np.ndarray | None = None,  # type: ignore
                       vpadding=0, hpadding=0, vstride=1, hstride=1,
                       vdilation=1, hdilation=1,
                       # biases originaly was called "biases_vector"
                       biases: np.ndarray | None = None,   # type: ignore
                       trans=False,
                       bn_running_mean: np.ndarray | None = None,  # type: ignore
                       bn_inv_std: np.ndarray | None = None,  # type: ignore
                       bn_gamma: np.ndarray | None = None,   # type: ignore
                       bn_beta: np.ndarray | None = None,   # type: ignore
                       relu=False):
        """
        Calls the appropriate convGemm function from libconvGemm.so to perform a
        matrix matrix multiplication with an implicit im2col.

        The matrix matrix product is in the form C = A * B, where:
            + A is the weights matrix,
            + B is the im2col(x) matrix, and
            + C is the out matrix.

        If the out vector is supplied, the xapplyBias function of the libconvGemm library will be called. This
        function sums each element of the out vector to all the elements in the corresponding output channel.

        Parameters
        ----------
        weights : array_like
            The weights matrix (kn x c x kh x kw).
        x : array_like
            The layers matrix (b x c x h x w).
        out : array_like
            An optional out matrix (kn x b*ho*wo). If provided, can be overwritten.
        vpadding : int
            The vertical padding to be applied to the x matrix.
        hpadding : int
            The horizontal padding to be applied to the x matrix.
        vstride : int
            The vertical stride.
        hstride : int
            The horizontal stride.
        vdilation : int
            The vertical dilation.
        hdilation : int
            The horizontal dilation.
        biases: array_like
            The out that have to be summed to all the elements in each output channel.
        trans: bool
            Perform the im2col(x) if False, or the im2colT(x) if True.

        Returns
        -------
        array_like
            The result of weights * im2col(x)
        """

        # Get matrices dimensions
        b, c, h, w = x.shape
        if not trans:
            kn, ck, kh, kw = weights.shape
            ho = (h + 2 * vpadding - vdilation * (kh - 1) - 1) // vstride + 1
            wo = (w + 2 * hpadding - hdilation * (kw - 1) - 1) // hstride + 1
            if out is None:
                out = np.zeros((b, kn, ho, wo), weights.dtype, order="C")
            else:
                out = out[:b, :]
                bb, knb, hob, wob = out.shape
                assert bb == b, "Batch size of the out must be the same as in the input!"
                assert knb == kn, "Number of filters in out must be the same as in the filter tensor!"
                assert hob == ho, "Biases image height must be the same as the output image height!"
                assert wob == wo, "Biases image width must be the same as the output image width!"
        else:
            assert out is not None, "If using the transposed convGemm, the out matrix must be supplied"
            kn, ck, kh, kw = out.shape
            bw, knw, ho, wo = weights.shape
            assert kn == knw, "Number of filters must be the same!"
            assert b == bw, "Batch size must be the same!"
        assert ck == c, "Number of channels in weights and x should be the same!"

        out: np.ndarray
        biases: np.ndarray
        bn_running_mean: np.ndarray
        bn_inv_std: np.ndarray
        bn_gamma: np.ndarray
        bn_beta: np.ndarray

        # Check that dtype is the same on all the matrices
        assert weights.dtype == x.dtype == out.dtype, \
            "All the matrices must have the same type of data!"
        assert weights.dtype == self.dtype, \
            "The input matrices must have the same type of data as the one specified when " \
            "this class was instantiated!"

        # Call the appropriate convGemm function from libconvGemm
        self.x_conv_gemm_nchw(ctypes.c_char(b'Y' if trans else b'N'),
                              ctypes.c_int(b), ctypes.c_int(c), ctypes.c_int(h), ctypes.c_int(w),
                              ctypes.c_int(kn), ctypes.c_int(kh), ctypes.c_int(kw),
                              ctypes.c_int(vpadding), ctypes.c_int(hpadding),
                              ctypes.c_int(vstride), ctypes.c_int(hstride),
                              ctypes.c_int(vdilation), ctypes.c_int(hdilation),
                              ctypes.c_void_p(weights.ctypes.data),
                              ctypes.c_void_p(x.ctypes.data),
                              ctypes.c_void_p(out.ctypes.data),
                              ctypes.c_void_p(None if biases is None else biases.ctypes.data),
                              ctypes.c_void_p(None if bn_running_mean is None else bn_running_mean.ctypes.data),
                              ctypes.c_void_p(None if bn_inv_std is None else bn_inv_std.ctypes.data),
                              ctypes.c_void_p(None if bn_gamma is None else bn_gamma.ctypes.data),
                              ctypes.c_void_p(None if bn_beta is None else bn_beta.ctypes.data), ctypes.c_bool(relu),
                              self.ac_pack, self.bc_pack)

        return out

    # TODO: Check for what is out used inside "x_conv_gemm_nhwc" (and set better varible names).
    def conv_gemm_nhwc(self, weights: np.ndarray, x: np.ndarray,
                       out: np.ndarray | None = None,  # type: ignore
                       vpadding=0, hpadding=0, vstride=1, hstride=1,
                       vdilation=1, hdilation=1,
                       biases: np.ndarray | None = None,   # type: ignore
                       trans=False,
                       bn_running_mean: np.ndarray | None = None,  # type: ignore
                       bn_inv_std: np.ndarray | None = None,  # type: ignore
                       bn_gamma: np.ndarray | None = None,   # type: ignore
                       bn_beta: np.ndarray | None = None,   # type: ignore
                       relu=False):

        # Get matrices dimensions
        b, h, w, c = x.shape
        if not trans:
            ck, kh, kw, kn = weights.shape
            ho = (h + 2 * vpadding - vdilation * (kh - 1) - 1) // vstride + 1
            wo = (w + 2 * hpadding - hdilation * (kw - 1) - 1) // hstride + 1
            if out is None:
                out = np.zeros((b, ho, wo, kn), weights.dtype, order="C")
            else:
                out = out[:b, :]
                bb, hob, wob, knb = out.shape
                assert bb == b, "Batch size of the out must be the same as in the input!"
                assert hob == ho, "Biases image height must be the same as the output image height!"
                assert wob == wo, "Biases image width must be the same as the output image width!"
                assert knb == kn, "Number of filters in out must be the same as in the filter tensor!"
        else:
            assert out is not None, "If using the transposed convGemm, the output matrix must be supplied"
            ck, kh, kw, kn = out.shape
            bw, ho, wo, knw = weights.shape
            assert kn == knw, "Number of filters must be the same!"
            assert b == bw, "Batch size must be the same!"
        assert ck == c, "Number of channels in weights and x should be the same!"

        out: np.ndarray
        biases: np.ndarray
        bn_running_mean: np.ndarray
        bn_inv_std: np.ndarray
        bn_gamma: np.ndarray
        bn_beta: np.ndarray

        # Check that dtype is the same on all the matrices
        assert weights.dtype == x.dtype == out.dtype, \
            "All the matrices must have the same type of data!"
        assert weights.dtype == self.dtype, \
            "The input matrices must have the same type of data as the one specified when " \
            "this class was instantiated!"

        # Call the appropriate convGemm function from libconvGemm
        self.x_conv_gemm_nhwc(ctypes.c_char(b'Y' if trans else b'N'),
                              ctypes.c_int(b), ctypes.c_int(h), ctypes.c_int(w), ctypes.c_int(c),
                              ctypes.c_int(kn), ctypes.c_int(kh), ctypes.c_int(kw),
                              ctypes.c_int(vpadding), ctypes.c_int(hpadding),
                              ctypes.c_int(vstride), ctypes.c_int(hstride),
                              ctypes.c_int(vdilation), ctypes.c_int(hdilation),
                              ctypes.c_void_p(weights.ctypes.data),
                              ctypes.c_void_p(x.ctypes.data),
                              ctypes.c_void_p(out.ctypes.data),
                              ctypes.c_void_p(None if biases is None else biases.ctypes.data),
                              ctypes.c_void_p(None if bn_running_mean is None else bn_running_mean.ctypes.data),
                              ctypes.c_void_p(None if bn_inv_std is None else bn_inv_std.ctypes.data),
                              ctypes.c_void_p(None if bn_gamma is None else bn_gamma.ctypes.data),
                              ctypes.c_void_p(None if bn_beta is None else bn_beta.ctypes.data), ctypes.c_bool(relu),
                              self.ac_pack, self.bc_pack)

        return out

    def deconv_gemm_nchw(self, weights: np.ndarray,
                         dy: np.ndarray,
                         dx: np.ndarray,
                         vpadding=0, hpadding=0,
                         vstride=1, hstride=1,
                         vdilation=1, hdilation=1):
        """
        Calls the appropriate deconv_gemm function from libconvGemm.so to perform
        an inplace matrix matrix multiplication and deconvolution:

            dx = col2im(weights_2D_T * dy_2D),

        where:
          * weights_2D_T is the weights matrix reshaped to 2D and transposed (c·kh·kw x kn),
          * dy_2D is the dy matrix transposed_1023 and reshaped to 2D (kn x b·ho·wo).

        Parameters
        ----------
        weights : array_like
            The weights matrix (kn x c x kh x kw).
        dy : array_like
            The dy matrix (b x kn x ho x wo).
        dx : array_like
            An empty dx matrix (b x c x h x w) that will be overwritten with col2im(* weights_2D_T * dy_2D).
        vpadding : int
            The vertical padding to be applied to the x matrix.
        hpadding : int
            The horizontal padding to be applied to the x matrix.
        vstride : int
            The vertical stride.
        hstride : int
            The horizontal stride.
        vdilation : int
            The vertical dilation.
        hdilation : int
            The horizontal dilation.

        Returns
        -------
        array_like
            The dx matrix.
        """

        # Get matrices dimensions
        kn, ck, kh, kw = weights.shape
        b2, kn2, ho, wo = dy.shape
        b, c, h, w = dx.shape
        assert kn == kn2, "Number of filters outputs in weights and dy should be the same!"
        assert b == b2, "Different batch size!"
        assert ck == c, "Number of channels in weights and x should be the same!"

        self.x_deconv_gemm_nchw(ctypes.c_int(b), ctypes.c_int(c), ctypes.c_int(h), ctypes.c_int(w),
                                ctypes.c_int(kn), ctypes.c_int(kh), ctypes.c_int(kw),
                                ctypes.c_int(vstride), ctypes.c_int(hstride),
                                ctypes.c_int(vpadding), ctypes.c_int(hpadding),
                                ctypes.c_int(vdilation), ctypes.c_int(hdilation),
                                ctypes.c_void_p(weights.ctypes.data),
                                ctypes.c_void_p(dy.ctypes.data),
                                ctypes.c_void_p(dx.ctypes.data),
                                self.ac_pack, self.bc_pack)

        return dx

    def deconv_gemm_nhwc(self, weights: np.ndarray,
                         dy: np.ndarray,
                         dx: np.ndarray,
                         vpadding=0, hpadding=0,
                         vstride=1, hstride=1,
                         vdilation=1, hdilation=1):

        ck, kh, kw, kn = weights.shape
        b2, ho, wo, kn2 = dy.shape
        b, h, w, c = dx.shape
        assert kn == kn2, "Number of filters outputs in weights and dy should be the same!"
        assert b == b2, "Different batch size!"
        assert ck == c, "Number of channels in weights and x should be the same!"

        self.x_deconv_gemm_nhwc(ctypes.c_int(b), ctypes.c_int(h), ctypes.c_int(w), ctypes.c_int(c),
                                ctypes.c_int(kn), ctypes.c_int(kh), ctypes.c_int(kw),
                                ctypes.c_int(vstride), ctypes.c_int(hstride),
                                ctypes.c_int(vpadding), ctypes.c_int(hpadding),
                                ctypes.c_int(vdilation), ctypes.c_int(hdilation),
                                ctypes.c_void_p(weights.ctypes.data),
                                ctypes.c_void_p(dy.ctypes.data), ctypes.c_void_p(dx.ctypes.data),
                                self.ac_pack, self.bc_pack)

        return dx


def __free__(pack):
    def find_msvcr():
        import re
        import sys
        exec_bytes = open(sys.executable, "rb").read()
        match = re.search("msvcr([0-9]+|t).dll", str(exec_bytes), re.IGNORECASE)
        return match.group(0)

    if platform.system() == 'Windows':
        libc = ctypes.cdll.LoadLibrary(find_msvcr())
    elif platform.system() == 'Linux':
        libc = ctypes.cdll.LoadLibrary('libc.so.6')
    elif platform.system == 'Darwin':
        libc = ctypes.cdll.LoadLibrary('libc.dylib')
    else:
        raise AssertionError("Don't know how to get to libc for a '{}' system".format(platform.system()))
    assert isinstance(pack, object)
    libc.free(pack)


def time_it_func(x: np.ndarray, w_c: np.ndarray, out: np.ndarray,
                 b: int, kn: int,
                 ho: int, wo: int, kh: int, kw: int,
                 vpadding: int, hpadding: int, vstride: int, hstride: int,
                 vdilation: int, hdilation: int,
                 ) -> int | float:

    res = np.zeros(((x.shape[0] * ho * wo), (x.shape[-1] * kh * kw)), dtype=x.dtype)
    im2col_nchw_cython(x, res,
                       kh, kw, ho, wo,
                       vpadding, hpadding, vstride, hstride,
                       vdilation, hdilation)
    res = res @ w_c
    res += out.reshape(b * ho * wo, kn)
    return res


def __usage_example__():
    # Imports for this usage example (not required otherwise)
    from timeit import timeit
    # Default parameters (1st layer AlexNet for Cifar10)
    b = 64  # Batch size
    c = 3  # Channels per layer
    h = 32  # Layers height
    w = 32  # Layers width
    kn = 64  # Number of filters
    kh = 3  # Filters weights height
    kw = 3  # Filters weights width
    vpadding = 1  # Vertical padding
    hpadding = 1  # Horizontal padding
    vstride = 2  # Vertical stride
    hstride = 2  # Horizontal stride
    vdilation = 1  # Vertical dilation
    hdilation = 1  # Horizontal dilation
    # Create weights, x, and out matrices from previous parameters. If no out
    # matrix is provided, a proper one filled with zeros will be automatically
    # created.
    weights = np.zeros((kn, c, kh, kw)).astype(np.float32, order='C')
    weights[0][0][0][0] = 1.89
    weights[1][1][1][1] = 3.0
    weights[2][2][2][2] = 4.0
    x = np.ones((b, c, h, w)).astype(np.float32, order='C')
    ho = (h + 2 * vpadding - vdilation * (kh - 1) - 1) // vstride + 1
    wo = (w + 2 * hpadding - hdilation * (kw - 1) - 1) // hstride + 1
    out = (np.ones((kn, b * ho * wo)) * 10).astype(np.float32, order='C')
    print("Using conv_gemm to compute alpha * weights * im2col(x) + beta * out...")
    conv_gemm = ConvGemm(debug=False)
    conv_gemm_result = conv_gemm.conv_gemm_nchw(weights, x,
                                                vpadding=vpadding, hpadding=hpadding,
                                                vstride=vstride, hstride=hstride,
                                                vdilation=vdilation, hdilation=hdilation,
                                                out=out.reshape(kn, b, ho, wo))
    print(conv_gemm_result)
    print("Sum: ", conv_gemm_result.sum())
    print()
    print("Using im2col and mm...")
    x_c = np.zeros((c * kh * kw, b * ho * wo))
    im2col_nchw_cython(x, x_c,
                       kh, kw, ho, wo,
                       vpadding, hpadding, vstride, hstride, vdilation, hdilation)
    w_c = weights.reshape(kn, -1)
    im2col_mm_result = (w_c @ x_c + out).reshape(kn, b, ho, wo).transpose(1, 0, 2, 3)
    print(im2col_mm_result)
    print("Sum: ", im2col_mm_result.sum())
    print("np.allclose: ", np.allclose(conv_gemm_result, im2col_mm_result))
    # print(conv_gemm_result - im2col_mm_result)
    # Times
    conv_gemm_t = timeit(lambda: conv_gemm.conv_gemm_nchw(weights, x,
                                                          vpadding=vpadding, hpadding=hpadding,
                                                          vstride=vstride, hstride=hstride,
                                                          vdilation=vdilation, hdilation=hdilation),
                         number=10) / 10
    print("Times")
    print("-----")
    print("conv_gemm time: {:.4f}".format(conv_gemm_t))
    im2col_t = timeit(lambda: time_it_func(x, w_c, out, b, kn, ho, wo, kh, kw, vpadding, hpadding, vstride, hstride, vdilation, hdilation),
                      number=10) / 10
    mm_t = timeit(lambda: w_c @ x_c + out, number=10) / 10
    print("im2col+mm time: {:.4f}  (im2col: {:.4f}  mm: {:.4f}".format(im2col_t + mm_t, im2col_t, mm_t))


if __name__ == "__main__":
    __usage_example__()
