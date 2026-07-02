"""
PyDTNN convGemm module

This module provides the ConvGemm class, which wraps the libconvGemm.so
library to perform efficient convolution operations using General Matrix
Multiply (GEMM) with implicit im2col/col2im transformations. It supports
both NCHW and NHWC data formats and includes functionalities for standard
convolutions and transposed convolutions (deconvolutions).
"""

import ctypes
import logging
import platform
import weakref

import numpy as np

from pydtnn.abstract.layerable import Layerable
from pydtnn.backends.cython.utils.im2col_nchw_cython import im2col_nchw_cython
from pydtnn.utils import load_library

__all__ = ("ConvGemm", "is_conv_gemm_available")

logger = logging.getLogger(__name__)


try:
    load_library("convGemm")
    is_conv_gemm_available = True
except Exception:
    is_conv_gemm_available = False


class ConvGemm:
    """
    Exposes the libconvGemm functions following the PyDTNN conventions.

    This class acts as a wrapper for the `libconvGemm.so` shared library,
    providing an interface to perform convolution operations efficiently
    by leveraging GEMM (General Matrix Multiply) with implicit im2col
    transformations. It supports both NCHW and NHWC data layouts and
    handles both forward and backward passes for convolutions and
    transposed convolutions (deconvolutions).

    Attributes
    ----------
    lib_cg : ctypes.CDLL or None
        The loaded `libconvGemm.so` library. It is shared across all instances
        of `ConvGemm`.
    dtype : np.dtype
        The data type (e.g., `np.float32`) used for matrix operations.
    ac_pack : ctypes.POINTER(ctypes.c_float)
        Pointer to an auxiliary buffer used internally by `libconvGemm`.
    bc_pack : ctypes.POINTER(ctypes.c_float)
        Pointer to another auxiliary buffer used internally by `libconvGemm`.
    debug : bool
        Flag to enable or disable debug logging.
    get_parent_layer : weakref.ref
        A weak reference to the parent layer, used for tracing purposes.
    x_conv_gemm_nhwc : ctypes.c_void_p
        Function pointer to the NHWC convolution GEMM implementation in `libconvGemm`.
    x_deconv_gemm_nhwc : ctypes.c_void_p
        Function pointer to the NHWC transposed convolution (deconvolution) GEMM
        implementation in `libconvGemm`.
    x_conv_gemm_nchw : ctypes.c_void_p
        Function pointer to the NCHW convolution GEMM implementation in `libconvGemm`.
    x_deconv_gemm_nchw : ctypes.c_void_p
        Function pointer to the NCHW transposed convolution (deconvolution) GEMM
        implementation in `libconvGemm`.

    Methods
    -------
    __init__(dtype, debug, parent_layer)
        Initializes the ConvGemm instance, loads the library, and allocates
        necessary internal buffers.
    __del__()
        Frees the allocated internal buffers when the instance is garbage collected.
    conv_gemm_nchw(...)
        Performs a convolution operation using the NCHW data format.
    conv_gemm_nhwc(...)
        Performs a convolution operation using the NHWC data format.
    deconv_gemm_nchw(...)
        Performs a transposed convolution (deconvolution) operation using the NCHW
        data format.
    deconv_gemm_nhwc(...)
        Performs a transposed convolution (deconvolution) operation using the NHWC
        data format.

    Examples
    --------
    See `__usage_example__()` method for an example of use. This example can be
    run with: 'python conv_gemm.py'

    Tests
    -----
    To perform the tests, run the following command from the current directory:
        python -m unittest tests.ConvGemmTestCase

    (see tests/conv_gemm.py for more instructions on testing)
    """

    lib_cg = None  # will link to the libconvGemm.so library

    def __init__(
        self,
        dtype: np.dtype = np.dtype(np.float32),
        debug: bool = False,
        parent_layer: Layerable | None = None,
    ) -> None:
        """
        Initializes the ConvGemm instance.

        Loads the `libconvGemm.so` library if it hasn't been loaded already
        and allocates the required auxiliary matrices (`ac_pack` and `bc_pack`)
        for internal use by the C/C++ library. It also selects the appropriate
        convolution GEMM functions based on the specified data type.

        Parameters
        ----------
        dtype : np.dtype, optional
            The element data type to be used for all matrices. Defaults to `np.float32`.
        debug : bool, optional
            If `True`, enables debug information printing. Defaults to `False`.
        parent_layer : object, optional
            A reference to the parent layer object, used for tracing purposes.
            Defaults to `None`.

        Raises
        ------
        MemoryError
            If the internal auxiliary buffers (`ac_pack` or `bc_pack`) cannot be
            allocated.
        TypeError
            If the specified `dtype` is not supported by the loaded `libconvGemm`
            library.
        """
        self.dtype = dtype
        if ConvGemm.lib_cg is None:
            ConvGemm.lib_cg = load_library("convGemm")
        assert self.lib_cg

        # Declare ac_pack and bc_pack and allocate space for them
        self.ac_pack = ctypes.POINTER(ctypes.c_float)()
        self.bc_pack = ctypes.POINTER(ctypes.c_float)()
        self.lib_cg.alloc_pack_buffs.restype = ctypes.c_int
        result = self.lib_cg.alloc_pack_buffs(
            ctypes.byref(self.ac_pack), ctypes.byref(self.bc_pack)
        )
        if result == 1:
            raise MemoryError("Could not allocate space for ac_pack or bc_pack!")
        # Debug
        self.debug = debug
        # Parent layer
        if parent_layer is not None:
            self.get_parent_layer = weakref.ref(parent_layer)
        # Choose the appropriate convGemm function depending on the architecture
        # and the data type being used
        if self.dtype == np.float32:
            self.x_conv_gemm_nhwc = self.lib_cg.sconvGemmNHWC
            self.x_deconv_gemm_nhwc = self.lib_cg.sconvGemmNHWC_back
            self.x_conv_gemm_nchw = self.lib_cg.sconvGemmNCHW
            self.x_deconv_gemm_nchw = self.lib_cg.sconvGemmNCHW_back
        else:
            raise TypeError(
                f"Type '{str(self.dtype)}' not supported by this version of libconvGemm!"
            )

    def __del__(self) -> None:
        """
        Frees the allocated internal auxiliary buffers.

        This method is called when the `ConvGemm` instance is garbage collected.
        It ensures that the memory allocated for `ac_pack` and `bc_pack` by
        `libconvGemm` is properly released.
        """
        try:
            # Assuming __free__ is available and handles platform-specific freeing
            __free__(self.ac_pack)
            __free__(self.bc_pack)
        except AttributeError:
            pass

    def conv_gemm_nchw(
        self,
        weights: np.ndarray,
        x: np.ndarray,
        # res originaly was called "biases"
        out: np.ndarray | None = None,  # type: ignore
        vpadding: int = 0,
        hpadding: int = 0,
        vstride: int = 1,
        hstride: int = 1,
        vdilation: int = 1,
        hdilation: int = 1,
        # biases originaly was called "biases_vector"
        biases: np.ndarray | None = None,  # type: ignore
        trans: bool = False,
        bn_running_mean: np.ndarray | None = None,  # type: ignore
        bn_inv_std: np.ndarray | None = None,  # type: ignore
        bn_gamma: np.ndarray | None = None,  # type: ignore
        bn_beta: np.ndarray | None = None,  # type: ignore
        relu: bool = False,
    ) -> np.ndarray:
        """
        Performs a convolution operation using the NCHW data format via GEMM.

        This method calls the `sconvGemmNCHW` function from `libconvGemm.so`
        to perform a convolution. It implicitly transforms the input tensor `x`
        into an im2col format and then performs a matrix multiplication with
        the `weights`. Optionally, it can apply biases, batch normalization,
        and ReLU activation.

        The core operation is equivalent to:
        `out = weights * im2col(x)`

        If `biases` are provided, they are added to the output channels.
        If batch normalization parameters (`bn_running_mean`, `bn_inv_std`,
        `bn_gamma`, `bn_beta`) are provided, they are applied after the
        convolution and bias addition. If `relu` is `True`, a ReLU activation
        is applied.

        Parameters
        ----------
        weights : np.ndarray
            The convolution kernel weights. Expected shape: `(kn, c, kh, kw)`,
            where `kn` is the number of output channels (filters), `c` is the
            number of input channels, `kh` is the kernel height, and `kw` is
            the kernel width.
        x : np.ndarray
            The input tensor. Expected shape: `(b, c, h, w)`, where `b` is the
            batch size, `c` is the number of input channels, `h` is the input
            height, and `w` is the input width.
        out : np.ndarray, optional
            An optional output tensor. If provided, it will be used to store the
            result. Its shape should be `(b, kn, ho, wo)`, where `ho` and `wo`
            are the output height and width. If `None`, a new tensor will be
            created.
        vpadding : int, optional
            The vertical padding to apply to the input `x`. Defaults to `0`.
        hpadding : int, optional
            The horizontal padding to apply to the input `x`. Defaults to `0`.
        vstride : int, optional
            The vertical stride for the convolution. Defaults to `1`.
        hstride : int, optional
            The horizontal stride for the convolution. Defaults to `1`.
        vdilation : int, optional
            The vertical dilation rate for the convolution. Defaults to `1`.
        hdilation : int, optional
            The horizontal dilation rate for the convolution. Defaults to `1`.
        biases : np.ndarray, optional
            An optional bias vector. If provided, its shape should be `(kn,)`
            and it will be added to each output channel. Defaults to `None`.
        trans : bool, optional
            If `False` (default), performs a standard convolution. If `True`,
            it implies a transposed convolution (deconvolution) operation,
            though this specific method is named `conv_gemm_nchw` and typically
            used for forward pass. The `trans` parameter might control internal
            behavior or be intended for a different function.
        bn_running_mean : np.ndarray, optional
            The running mean for batch normalization. Defaults to `None`.
        bn_inv_std : np.ndarray, optional
            The inverse standard deviation for batch normalization. Defaults to `None`.
        bn_gamma : np.ndarray, optional
            The gamma (scale) parameter for batch normalization. Defaults to `None`.
        bn_beta : np.ndarray, optional
            The beta (shift) parameter for batch normalization. Defaults to `None`.
        relu : bool, optional
            If `True`, applies the ReLU activation function to the output.
            Defaults to `False`.

        Returns
        -------
        np.ndarray
            The resulting output tensor after the convolution operation. Its shape
            will be `(b, kn, ho, wo)`.

        Raises
        ------
        AssertionError
            If input tensor dimensions or data types are inconsistent or do not
            match expectations.
        TypeError
            If the input matrices do not have the same data type as specified
            during `ConvGemm` instantiation.
        """

        # Get matrices dimensions
        b, c, h, w = x.shape
        if not trans:
            kn, ck, kh, kw = weights.shape
            ho = (h + 2 * vpadding - vdilation * (kh - 1) - 1) // vstride + 1
            wo = (w + 2 * hpadding - hdilation * (kw - 1) - 1) // hstride + 1
            if out is None:
                out = np.zeros((b, kn, ho, wo), weights.dtype)
            else:
                out = out[:b, :]
                bb, knb, hob, wob = out.shape
                assert bb == b, "Batch size of the out must be the same as in the input!"
                assert knb == kn, (
                    "Number of filters in out must be the same as in the filter tensor!"
                )
                assert hob == ho, "Biases image height must be the same as the output image height!"
                assert wob == wo, "Biases image width must be the same as the output image width!"
        else:
            # This branch seems to be for transposed convolution logic, but the method name is conv_gemm_nchw.
            # Assuming 'trans' might be used for backward pass or specific GEMM configurations.
            assert out is not None, (
                "If using the transposed convGemm, the out matrix must be supplied"
            )
            kn, ck, kh, kw = out.shape  # Assuming out shape is (kn, b, ho, wo) for transposed
            bw, knw, ho, wo = (
                weights.shape
            )  # Assuming weights shape is (c, kn, kh, kw) for transposed
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
        assert weights.dtype == x.dtype == out.dtype, (
            "All the matrices must have the same type of data!"
        )
        assert weights.dtype == self.dtype, (
            "The input matrices must have the same type of data as the one specified when this"
            " class was instantiated!"
        )

        # Call the appropriate convGemm function from libconvGemm
        self.x_conv_gemm_nchw(
            ctypes.c_char(b"Y" if trans else b"N"),
            ctypes.c_int(b),
            ctypes.c_int(c),
            ctypes.c_int(h),
            ctypes.c_int(w),
            ctypes.c_int(kn),
            ctypes.c_int(kh),
            ctypes.c_int(kw),
            ctypes.c_int(vpadding),
            ctypes.c_int(hpadding),
            ctypes.c_int(vstride),
            ctypes.c_int(hstride),
            ctypes.c_int(vdilation),
            ctypes.c_int(hdilation),
            ctypes.c_void_p(weights.ctypes.data),
            ctypes.c_void_p(x.ctypes.data),
            ctypes.c_void_p(out.ctypes.data),
            ctypes.c_void_p(None if biases is None else biases.ctypes.data),
            ctypes.c_void_p(None if bn_running_mean is None else bn_running_mean.ctypes.data),
            ctypes.c_void_p(None if bn_inv_std is None else bn_inv_std.ctypes.data),
            ctypes.c_void_p(None if bn_gamma is None else bn_gamma.ctypes.data),
            ctypes.c_void_p(None if bn_beta is None else bn_beta.ctypes.data),
            ctypes.c_bool(relu),
            self.ac_pack,
            self.bc_pack,
        )

        return out

    # TODO: Check for what is out used inside "x_conv_gemm_nhwc" (and set better varible names).
    def conv_gemm_nhwc(
        self,
        weights: np.ndarray,
        x: np.ndarray,
        out: np.ndarray | None = None,  # type: ignore
        vpadding: int = 0,
        hpadding: int = 0,
        vstride: int = 1,
        hstride: int = 1,
        vdilation: int = 1,
        hdilation: int = 1,
        biases: np.ndarray | None = None,  # type: ignore
        trans: bool = False,
        bn_running_mean: np.ndarray | None = None,  # type: ignore
        bn_inv_std: np.ndarray | None = None,  # type: ignore
        bn_gamma: np.ndarray | None = None,  # type: ignore
        bn_beta: np.ndarray | None = None,  # type: ignore
        relu: bool = False,
    ) -> np.ndarray:
        """
        Performs a convolution operation using the NHWC data format via GEMM.

        This method calls the `sconvGemmNHWC` function from `libconvGemm.so`
        to perform a convolution. It implicitly transforms the input tensor `x`
        into an im2col format and then performs a matrix multiplication with
        the `weights`. Optionally, it can apply biases, batch normalization,
        and ReLU activation.

        The core operation is equivalent to:
        `out = weights * im2col(x)`

        If `biases` are provided, they are added to the output channels.
        If batch normalization parameters (`bn_running_mean`, `bn_inv_std`,
        `bn_gamma`, `bn_beta`) are provided, they are applied after the
        convolution and bias addition. If `relu` is `True`, a ReLU activation
        is applied.

        Parameters
        ----------
        weights : np.ndarray
            The convolution kernel weights. Expected shape: `(ck, kh, kw, kn)`,
            where `ck` is the number of input channels, `kh` is the kernel height,
            `kw` is the kernel width, and `kn` is the number of output channels
            (filters).
        x : np.ndarray
            The input tensor. Expected shape: `(b, h, w, c)`, where `b` is the
            batch size, `h` is the input height, `w` is the input width, and `c`
            is the number of input channels.
        out : np.ndarray, optional
            An optional output tensor. If provided, it will be used to store the
            result. Its shape should be `(b, ho, wo, kn)`, where `ho` and `wo`
            are the output height and width. If `None`, a new tensor will be
            created.
        vpadding : int, optional
            The vertical padding to apply to the input `x`. Defaults to `0`.
        hpadding : int, optional
            The horizontal padding to apply to the input `x`. Defaults to `0`.
        vstride : int, optional
            The vertical stride for the convolution. Defaults to `1`.
        hstride : int, optional
            The horizontal stride for the convolution. Defaults to `1`.
        vdilation : int, optional
            The vertical dilation rate for the convolution. Defaults to `1`.
        hdilation : int, optional
            The horizontal dilation rate for the convolution. Defaults to `1`.
        biases : np.ndarray, optional
            An optional bias vector. If provided, its shape should be `(kn,)`
            and it will be added to each output channel. Defaults to `None`.
        trans : bool, optional
            If `False` (default), performs a standard convolution. If `True`,
            it implies a transposed convolution (deconvolution) operation,
            though this specific method is named `conv_gemm_nhwc` and typically
            used for forward pass. The `trans` parameter might control internal
            behavior or be intended for a different function.
        bn_running_mean : np.ndarray, optional
            The running mean for batch normalization. Defaults to `None`.
        bn_inv_std : np.ndarray, optional
            The inverse standard deviation for batch normalization. Defaults to `None`.
        bn_gamma : np.ndarray, optional
            The gamma (scale) parameter for batch normalization. Defaults to `None`.
        bn_beta : np.ndarray, optional
            The beta (shift) parameter for batch normalization. Defaults to `None`.
        relu : bool, optional
            If `True`, applies the ReLU activation function to the output.
            Defaults to `False`.

        Returns
        -------
        np.ndarray
            The resulting output tensor after the convolution operation. Its shape
            will be `(b, ho, wo, kn)`.

        Raises
        ------
        AssertionError
            If input tensor dimensions or data types are inconsistent or do not
            match expectations.
        TypeError
            If the input matrices do not have the same data type as specified
            during `ConvGemm` instantiation.
        """

        # Get matrices dimensions
        b, h, w, c = x.shape
        if not trans:
            ck, kh, kw, kn = weights.shape
            ho = (h + 2 * vpadding - vdilation * (kh - 1) - 1) // vstride + 1
            wo = (w + 2 * hpadding - hdilation * (kw - 1) - 1) // hstride + 1
            if out is None:
                out = np.zeros((b, ho, wo, kn), weights.dtype)
            else:
                out = out[:b, :]
                bb, hob, wob, knb = out.shape
                assert bb == b, "Batch size of the out must be the same as in the input!"
                assert hob == ho, "Biases image height must be the same as the output image height!"
                assert wob == wo, "Biases image width must be the same as the output image width!"
                assert knb == kn, (
                    "Number of filters in out must be the same as in the filter tensor!"
                )
        else:
            # This branch seems to be for transposed convolution logic, but the method name is conv_gemm_nhwc.
            # Assuming 'trans' might be used for backward pass or specific GEMM configurations.
            assert out is not None, (
                "If using the transposed convGemm, the output matrix must be supplied"
            )
            ck, kh, kw, kn = out.shape  # Assuming out shape is (ho, wo, kn) for transposed
            bw, ho, wo, knw = (
                weights.shape
            )  # Assuming weights shape is (h, w, c, kn) for transposed
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
        assert weights.dtype == x.dtype == out.dtype, (
            "All the matrices must have the same type of data!"
        )
        assert weights.dtype == self.dtype, (
            "The input matrices must have the same type of data as the one specified when this"
            " class was instantiated!"
        )

        # Call the appropriate convGemm function from libconvGemm
        self.x_conv_gemm_nhwc(
            ctypes.c_char(b"Y" if trans else b"N"),
            ctypes.c_int(b),
            ctypes.c_int(h),
            ctypes.c_int(w),
            ctypes.c_int(c),
            ctypes.c_int(kn),
            ctypes.c_int(kh),
            ctypes.c_int(kw),
            ctypes.c_int(vpadding),
            ctypes.c_int(hpadding),
            ctypes.c_int(vstride),
            ctypes.c_int(hstride),
            ctypes.c_int(vdilation),
            ctypes.c_int(hdilation),
            ctypes.c_void_p(weights.ctypes.data),
            ctypes.c_void_p(x.ctypes.data),
            ctypes.c_void_p(out.ctypes.data),
            ctypes.c_void_p(None if biases is None else biases.ctypes.data),
            ctypes.c_void_p(None if bn_running_mean is None else bn_running_mean.ctypes.data),
            ctypes.c_void_p(None if bn_inv_std is None else bn_inv_std.ctypes.data),
            ctypes.c_void_p(None if bn_gamma is None else bn_gamma.ctypes.data),
            ctypes.c_void_p(None if bn_beta is None else bn_beta.ctypes.data),
            ctypes.c_bool(relu),
            self.ac_pack,
            self.bc_pack,
        )

        return out

    def deconv_gemm_nchw(
        self,
        weights: np.ndarray,
        dy: np.ndarray,
        dx: np.ndarray,
        vpadding: int = 0,
        hpadding: int = 0,
        vstride: int = 1,
        hstride: int = 1,
        vdilation: int = 1,
        hdilation: int = 1,
    ) -> np.ndarray:
        """
        Performs a transposed convolution (deconvolution) operation using NCHW format.

        This method calls the `sconvGemmNCHW_back` function from `libconvGemm.so`
        to compute the gradient with respect to the input (`dx`) in a transposed
        convolution manner. It effectively performs `dx = col2im(weights_T * dy)`.

        Parameters
        ----------
        weights : np.ndarray
            The convolution kernel weights. Expected shape: `(kn, c, kh, kw)`,
            where `kn` is the number of output channels (filters), `c` is the
            number of input channels, `kh` is the kernel height, and `kw` is
            the kernel width.
        dy : np.ndarray
            The gradient tensor from the subsequent layer. Expected shape:
            `(b, kn, ho, wo)`, where `b` is the batch size, `kn` is the number
            of output channels (filters), `ho` is the output height, and `wo`
            is the output width.
        dx : np.ndarray
            An output tensor to store the computed gradient with respect to the
            input. It will be overwritten. Expected shape: `(b, c, h, w)`, where
            `b` is the batch size, `c` is the number of input channels, `h` is
            the input height, and `w` is the input width.
        vpadding : int, optional
            The vertical padding that was applied to the original input `x`
            during the forward pass. Defaults to `0`.
        hpadding : int, optional
            The horizontal padding that was applied to the original input `x`
            during the forward pass. Defaults to `0`.
        vstride : int, optional
            The vertical stride used in the forward pass. Defaults to `1`.
        hstride : int, optional
            The horizontal stride used in the forward pass. Defaults to `1`.
        vdilation : int, optional
            The vertical dilation rate used in the forward pass. Defaults to `1`.
        hdilation : int, optional
            The horizontal dilation rate used in the forward pass. Defaults to `1`.

        Returns
        -------
        np.ndarray
            The computed gradient tensor `dx`.

        Raises
        ------
        AssertionError
            If input tensor dimensions are inconsistent.
        """

        # Get matrices dimensions
        kn, ck, kh, kw = weights.shape
        b2, kn2, ho, wo = dy.shape
        b, c, h, w = dx.shape
        assert kn == kn2, "Number of filters outputs in weights and dy should be the same!"
        assert b == b2, "Different batch size!"
        assert ck == c, "Number of channels in weights and x should be the same!"

        self.x_deconv_gemm_nchw(
            ctypes.c_int(b),
            ctypes.c_int(c),
            ctypes.c_int(h),
            ctypes.c_int(w),
            ctypes.c_int(kn),
            ctypes.c_int(kh),
            ctypes.c_int(kw),
            ctypes.c_int(vstride),
            ctypes.c_int(hstride),
            ctypes.c_int(vpadding),
            ctypes.c_int(hpadding),
            ctypes.c_int(vdilation),
            ctypes.c_int(hdilation),
            ctypes.c_void_p(weights.ctypes.data),
            ctypes.c_void_p(dy.ctypes.data),
            ctypes.c_void_p(dx.ctypes.data),
            self.ac_pack,
            self.bc_pack,
        )

        return dx

    def deconv_gemm_nhwc(
        self,
        weights: np.ndarray,
        dy: np.ndarray,
        dx: np.ndarray,
        vpadding: int = 0,
        hpadding: int = 0,
        vstride: int = 1,
        hstride: int = 1,
        vdilation: int = 1,
        hdilation: int = 1,
    ) -> np.ndarray:
        """
        Performs a transposed convolution (deconvolution) operation using NHWC format.

        This method calls the `sconvGemmNHWC_back` function from `libconvGemm.so`
        to compute the gradient with respect to the input (`dx`) in a transposed
        convolution manner. It effectively performs `dx = col2im(weights_T * dy)`.

        Parameters
        ----------
        weights : np.ndarray
            The convolution kernel weights. Expected shape: `(ck, kh, kw, kn)`,
            where `ck` is the number of input channels, `kh` is the kernel height,
            `kw` is the kernel width, and `kn` is the number of output channels
            (filters).
        dy : np.ndarray
            The gradient tensor from the subsequent layer. Expected shape:
            `(b, ho, wo, kn)`, where `b` is the batch size, `ho` is the output
            height, `wo` is the output width, and `kn` is the number of output
            channels (filters).
        dx : np.ndarray
            An output tensor to store the computed gradient with respect to the
            input. It will be overwritten. Expected shape: `(b, h, w, c)`, where
            `b` is the batch size, `h` is the input height, `w` is the input width,
            and `c` is the number of input channels.
        vpadding : int, optional
            The vertical padding that was applied to the original input `x`
            during the forward pass. Defaults to `0`.
        hpadding : int, optional
            The horizontal padding that was applied to the original input `x`
            during the forward pass. Defaults to `0`.
        vstride : int, optional
            The vertical stride used in the forward pass. Defaults to `1`.
        hstride : int, optional
            The horizontal stride used in the forward pass. Defaults to `1`.
        vdilation : int, optional
            The vertical dilation rate used in the forward pass. Defaults to `1`.
        hdilation : int, optional
            The horizontal dilation rate used in the forward pass. Defaults to `1`.

        Returns
        -------
        np.ndarray
            The computed gradient tensor `dx`.

        Raises
        ------
        AssertionError
            If input tensor dimensions are inconsistent.
        """

        ck, kh, kw, kn = weights.shape
        b2, ho, wo, kn2 = dy.shape
        b, h, w, c = dx.shape
        assert kn == kn2, "Number of filters outputs in weights and dy should be the same!"
        assert b == b2, "Different batch size!"
        assert ck == c, "Number of channels in weights and x should be the same!"

        self.x_deconv_gemm_nhwc(
            ctypes.c_int(b),
            ctypes.c_int(h),
            ctypes.c_int(w),
            ctypes.c_int(c),
            ctypes.c_int(kn),
            ctypes.c_int(kh),
            ctypes.c_int(kw),
            ctypes.c_int(vstride),
            ctypes.c_int(hstride),
            ctypes.c_int(vpadding),
            ctypes.c_int(hpadding),
            ctypes.c_int(vdilation),
            ctypes.c_int(hdilation),
            ctypes.c_void_p(weights.ctypes.data),
            ctypes.c_void_p(dy.ctypes.data),
            ctypes.c_void_p(dx.ctypes.data),
            self.ac_pack,
            self.bc_pack,
        )

        return dx


def __free__(pack: ctypes._Pointer) -> None:
    """
    Frees a memory buffer allocated by `libc` on different platforms.

    This utility function is used to release memory allocated by C libraries,
    typically for auxiliary buffers used by `libconvGemm`. It dynamically
    loads the appropriate C standard library (`libc`) based on the operating
    system and calls its `free` function.

    Parameters
    ----------
    pack : ctypes.POINTER
        A pointer to the memory buffer to be freed.

    Raises
    ------
    AssertionError
        If the operating system is not supported or if `libc` cannot be found.
    """

    def find_msvcr():
        import re
        import sys

        exec_bytes = open(sys.executable, "rb").read()
        match = re.search("msvcr([0-9]+|t).dll", str(exec_bytes), re.IGNORECASE)
        assert match, "MSVCR not found!"
        return match.group(0)

    if platform.system() == "Windows":
        libc = ctypes.cdll.LoadLibrary(find_msvcr())
    elif platform.system() == "Linux":
        libc = ctypes.cdll.LoadLibrary("libc.so.6")
    elif platform.system == "Darwin":
        libc = ctypes.cdll.LoadLibrary("libc.dylib")
    else:
        raise AssertionError(
            "Don't know how to get to libc for a '{}' system".format(platform.system())
        )
    assert isinstance(pack, object)
    libc.free(pack)


def time_it_func(
    x: np.ndarray,
    w_c: np.ndarray,
    out: np.ndarray,
    b: int,
    kn: int,
    ho: int,
    wo: int,
    kh: int,
    kw: int,
    vpadding: int,
    hpadding: int,
    vstride: int,
    hstride: int,
    vdilation: int,
    hdilation: int,
) -> int | float:
    """
    Helper function to perform convolution using im2col and matrix multiplication for timing.

    This function is likely used for benchmarking or comparison purposes. It
    first transforms the input tensor `x` into an im2col format using
    `im2col_nchw_cython`, then performs a matrix multiplication with the
    weights `w_c`, and finally adds the `out` tensor (presumably biases or
    an initial output).

    Parameters
    ----------
    x : np.ndarray
        The input tensor.
    w_c : np.ndarray
        The weights matrix, likely reshaped for GEMM.
    out : np.ndarray
        An auxiliary tensor, possibly for biases or initial output.
    b : int
        Batch size.
    kn : int
        Number of output channels (filters).
    ho : int
        Output height.
    wo : int
        Output width.
    kh : int
        Kernel height.
    kw : int
        Kernel width.
    vpadding : int
        Vertical padding.
    hpadding : int
        Horizontal padding.
    vstride : int
        Vertical stride.
    hstride : int
        Horizontal stride.
    vdilation : int
        Vertical dilation.
    hdilation : int
        Horizontal dilation.

    Returns
    -------
    int or float
        The result of the operation, likely a numerical value or array.
        The return type hint `int | float` seems potentially inaccurate given
        the operations performed; it might return a numpy array.
    """

    res = np.zeros(((x.shape[0] * ho * wo), (x.shape[-1] * kh * kw)), dtype=x.dtype)
    im2col_nchw_cython(
        x,
        res,  # type: ignore
        kh,
        kw,
        ho,
        wo,
        vpadding,
        hpadding,
        vstride,
        hstride,
        vdilation,
        hdilation,
    )
    res = res @ w_c
    res += out.reshape(b * ho * wo, kn)
    return res  # type: ignore


def __usage_example__() -> None:
    """
    Provides a usage example for the `ConvGemm` class.

    This function demonstrates how to instantiate and use the `ConvGemm` class
    for performing NCHW convolutions. It sets up sample input tensors (`weights`,
    `x`), defines convolution parameters, performs the convolution using
    `conv_gemm.conv_gemm_nchw`, and compares the result with a manual
    implementation using `im2col_nchw_cython` and standard NumPy matrix
    multiplication. It also includes basic timing comparisons.

    The example can be run directly by executing the script.
    """
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
    weights = np.zeros((kn, c, kh, kw)).astype(np.float32, order="C")
    weights[0][0][0][0] = 1.89
    weights[1][1][1][1] = 3.0
    weights[2][2][2][2] = 4.0
    x = np.ones((b, c, h, w)).astype(np.float32, order="C")
    ho = (h + 2 * vpadding - vdilation * (kh - 1) - 1) // vstride + 1
    wo = (w + 2 * hpadding - hdilation * (kw - 1) - 1) // hstride + 1
    out = (np.ones((kn, b * ho * wo)) * 10).astype(np.float32, order="C")
    logger.info("Using conv_gemm to compute alpha * weights * im2col(x) + beta * out...")
    conv_gemm = ConvGemm(debug=False)
    conv_gemm_result = conv_gemm.conv_gemm_nchw(
        weights,
        x,
        vpadding=vpadding,
        hpadding=hpadding,
        vstride=vstride,
        hstride=hstride,
        vdilation=vdilation,
        hdilation=hdilation,
        out=out.reshape(kn, b, ho, wo),
    )
    logger.info(
        "\n".join(
            [str(conv_gemm_result), f"Sum: {conv_gemm_result.sum()}", "", "Using im2col and mm..."]
        )
    )
    x_c = np.zeros((c * kh * kw, b * ho * wo))
    im2col_nchw_cython(
        x,
        x_c,  # type: ignore
        kh,
        kw,
        ho,
        wo,
        vpadding,
        hpadding,
        vstride,
        hstride,
        vdilation,
        hdilation,
    )
    w_c = weights.reshape(kn, -1)
    im2col_mm_result = (w_c @ x_c + out).reshape(kn, b, ho, wo).transpose(1, 0, 2, 3)
    logger.info(
        "\n".join(
            [
                str(im2col_mm_result),
                f"Sum: {im2col_mm_result.sum()}",
                f"np.allclose: {np.allclose(conv_gemm_result, im2col_mm_result)}",
            ]
        )
    )
    # print(conv_gemm_result - im2col_mm_result)
    # Times
    conv_gemm_t = (
        timeit(
            lambda: conv_gemm.conv_gemm_nchw(
                weights,
                x,
                vpadding=vpadding,
                hpadding=hpadding,
                vstride=vstride,
                hstride=hstride,
                vdilation=vdilation,
                hdilation=hdilation,
            ),
            number=10,
        )
        / 10
    )
    logger.info("\n".join(["Times", "-----", "conv_gemm time: {:.4f}".format(conv_gemm_t)]))
    im2col_t = (
        timeit(
            lambda: time_it_func(
                x,
                w_c,
                out,
                b,
                kn,
                ho,
                wo,
                kh,
                kw,
                vpadding,
                hpadding,
                vstride,
                hstride,
                vdilation,
                hdilation,
            ),
            number=10,
        )
        / 10
    )
    mm_t = timeit(lambda: w_c @ x_c + out, number=10) / 10
    logger.info(
        "im2col+mm time: {:.4f}  (im2col: {:.4f}  mm: {:.4f}".format(
            im2col_t + mm_t, im2col_t, mm_t
        )
    )


if __name__ == "__main__":
    __usage_example__()
