"""
PyDTNN convWinograd module

This module provides an implementation of the Winograd convolution algorithm,
leveraging optimized C/C++ libraries for performance. It supports different
tensor formats (NCHW, NHWC) and aims to provide a faster alternative to
standard convolution implementations for specific kernel and stride configurations.
"""

import ctypes
import logging
import math
import platform
import weakref
from collections import defaultdict
from functools import partial
from typing import Any, Callable

import numpy as np

from pydtnn.abstract.layerable import Layerable
from pydtnn.backends.cython.utils.im2col_nchw_cython import im2col_nchw_cython
from pydtnn.backends.cython.utils.im2row_nhwc_cython import im2row_nhwc_cython
from pydtnn.utils import load_library
from pydtnn.utils.best_of.best_of import BestOf
from pydtnn.utils.tensor import TensorFormat, decode_shape, encode_shape

__all__ = ("ConvWinograd", "is_conv_winograd_available")

logger = logging.getLogger(__name__)


try:
    load_library("convwinograd")
    is_conv_winograd_available = True
except Exception:
    is_conv_winograd_available = False


class ConvWinograd:
    """
    Exposes the libconvWinograd functions following the PyDTNN conventions.

    This class acts as a wrapper around the compiled Winograd convolution library.
    It selects the appropriate optimized routine based on the input parameters
    (kernel size, strides, dilation, data type, and tensor format) and provides
    a unified interface for performing Winograd convolutions.

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

    lib_cw = None  # will link to the libconvwinograd.so library

    def winograd_workspace_alloc_pre(
        self, m: int, r: int, k: int, c: int
    ) -> tuple[np.ndarray, ctypes._Pointer[ctypes.c_float]]:
        """
        Allocates workspace memory for the Winograd pre-processing step.

        Parameters
        ----------
        m: int
            Winograd transform parameter 'm'.
        r: int
            Winograd transform parameter 'r'.
        k: int
            Number of output channels.
        c: int
            Number of input channels.

        Returns
        -------
        tuple
            A tuple containing a dummy numpy array and a ctypes pointer to the allocated memory.
        """
        _u = ctypes.POINTER(ctypes.c_float)()
        self.conv_winograd_workspace_alloc_pre(
            ctypes.c_uint(m),
            ctypes.c_uint(r),
            ctypes.c_uint(k),
            ctypes.c_uint(c),
            ctypes.byref(_u),  # pyright: ignore[reportArgumentType] (revise)
        )
        return np.array([False]), _u

    def winograd_workspace_alloc_kernel(
        self,
        m: int,
        r: int,
        n: int,
        k: int,
        c: int,
        hi: int,
        wi: int,
        kh: int,
        kw: int,
        vpadding: int,
        hpadding: int,
    ) -> tuple[ctypes._Pointer[ctypes.c_float], ctypes._Pointer[ctypes.c_float]]:
        """
        Allocates workspace memory for the Winograd kernel execution step.

        Parameters
        ----------
        m: int
            Winograd transform parameter 'm'.
        r: int
            Winograd transform parameter 'r'.
        n: int
            Batch size.
        k: int
            Number of output channels.
        c: int
            Number of input channels.
        hi: int
            Input height.
        wi: int
            Input width.
        kh: int
            Kernel height.
        kw: int
            Kernel width.
        vpadding: int
            Vertical padding.
        hpadding: int
            Horizontal padding.

        Returns
        -------
        tuple
            A tuple containing two ctypes pointers to the allocated memory (_v and _m).
        """
        _v = ctypes.POINTER(ctypes.c_float)()
        _m = ctypes.POINTER(ctypes.c_float)()
        self.conv_winograd_workspace_alloc_kernel(
            ctypes.c_uint(m),
            ctypes.c_uint(r),
            ctypes.c_uint(n),
            ctypes.c_uint(k),
            ctypes.c_uint(c),
            ctypes.c_uint(hi),
            ctypes.c_uint(wi),
            ctypes.c_uint(kh),
            ctypes.c_uint(kw),
            ctypes.c_uint(vpadding),
            ctypes.c_uint(hpadding),
            ctypes.byref(_v),  # pyright: ignore[reportArgumentType]
            ctypes.byref(_m),  # pyright: ignore[reportArgumentType]
        )
        return _v, _m

    def register_winograd_function(
        self, m: int, r: int, g: np.ndarray, bt: np.ndarray, at: np.ndarray
    ) -> None:
        """
        Registers available Winograd routines for a given Winograd transform size (m, r).

        It attempts to find optimized C/C++ routines in the loaded library
        for the current architecture and data type. If no optimized routine
        is found, it falls back to a NumPy implementation.

        Parameters
        ----------
        m: int
            The 'm' parameter of the Winograd transform (output tile size).
        r: int
            The 'r' parameter of the Winograd transform (input tile size).
        g: np.ndarray
            The transformation matrix G for the input data.
        bt: np.ndarray
            The transformation matrix B_T for the input data.
        at: np.ndarray
            The transformation matrix A_T for the output data.
        """
        # choose the appropriate convWinograd function depending on the
        # architecture and the data type being used
        if platform.machine() == "aarch64":
            if self.dtype == np.float32:
                routine_names = [
                    ("neon", f"conv_winograd_{m}x{m}_{r}x{r}_neon_fp32_{self.tensor_format}")
                ]
            else:
                raise NotImplementedError(
                    f"Type {str(self.dtype)} not supported by this version of libconvWinograd!"
                )
        elif platform.machine() == "x86_64":
            if self.dtype == np.float32:
                routine_names = [
                    (intr, f"conv_winograd_{m}x{m}_{r}x{r}_{intr}_fp32_{self.tensor_format}")
                    for intr in ["native", "sse", "avx", "avx512"]
                ]
            else:
                raise NotImplementedError(
                    f"Type {str(self.dtype)} not supported by this version of libconvWinograd!"
                )
        else:
            raise NotImplementedError(f"Platform '{str(platform.machine())}' not yet supported")

        funcs = []
        for rn in routine_names:
            try:
                funcs.append(
                    (
                        rn[0],
                        (
                            self._conv_winograd_c,
                            getattr(self.__class__.lib_cw, f"{rn[1]}_pre"),
                            getattr(self.__class__.lib_cw, f"{rn[1]}_kernel"),
                        ),
                    )
                )
            except AttributeError:
                pass
        if not funcs:
            logger.warning("Winograd routine not found. Fallback to numpy version!")
            funcs = [("numpy", (self._conv_winograd_numpy, None, None))]

        for intr, f in funcs:
            assert f[1], f"{f[0]} missing pre function"
            assert f[2], f"{f[0]} missing kernel function"
            self.alternatives[r].append(
                (f"cw{m}{r}{intr}", partial(f[0], m, r, g, bt, at, f[1], f[2]))
            )

    def register_winograd_function_3x3_2x2(self) -> None:
        """Register 3x3 output from 2x2 input winograd function"""
        m, r = 3, 2
        self.register_winograd_function(
            m,
            r,
            g=np.array(
                [[1, 0], [1.0 / 2.0, 1.0 / 2.0], [1.0 / 2.0, -1.0 / 2.0], [0, 1]],
                dtype=self.dtype,
            ),
            bt=np.array(
                [[1, 0, -1, 0], [0, 1, 1, 0], [0, -1, 1, 0], [0, -1, 0, 1]], dtype=self.dtype
            ),
            at=np.array([[1, 1, 1, 0], [0, 1, -1, 0], [0, 1, 1, 1]], dtype=self.dtype),
        )

    def register_winograd_function_2x2_3x3(self) -> None:
        """Register 2x2 output from 3x3 input winograd function"""
        m, r = 2, 3
        self.register_winograd_function(
            m,
            r,
            g=np.array(
                [
                    [1, 0, 0],
                    [1.0 / 2.0, 1.0 / 2.0, 1.0 / 2.0],
                    [1.0 / 2.0, -1.0 / 2.0, 1.0 / 2.0],
                    [0, 0, 1],
                ],
                dtype=self.dtype,
            ),
            bt=np.array(
                [[1, 0, -1, 0], [0, 1, 1, 0], [0, -1, 1, 0], [0, 1, 0, -1]], dtype=self.dtype
            ),
            at=np.array([[1, 1, 1, 0], [0, 1, -1, -1]], dtype=self.dtype),
        )

    def register_winograd_function_4x4_3x3(self) -> None:
        """Register 4x4 output from 3x3 input winograd function"""
        m, r = 4, 3
        self.register_winograd_function(
            m,
            r,
            g=np.array(
                [
                    [1.0 / 4.0, 0, 0],
                    [-1.0 / 6.0, -1.0 / 6.0, -1.0 / 6.0],
                    [-1.0 / 6.0, 1.0 / 6.0, -1.0 / 6.0],
                    [1.0 / 24.0, 1.0 / 12.0, 1.0 / 6.0],
                    [1.0 / 24.0, -1.0 / 12.0, 1.0 / 6.0],
                    [0, 0, 1],
                ],
                dtype=self.dtype,
            ),
            bt=np.array(
                [
                    [4, 0, -5, 0, 1, 0],
                    [0, -4, -4, 1, 1, 0],
                    [0, 4, -4, -1, 1, 0],
                    [0, -2, -1, 2, 1, 0],
                    [0, 2, -1, -2, 1, 0],
                    [0, 4, 0, -5, 0, 1],
                ],
                dtype=self.dtype,
            ),
            at=np.array(
                [
                    [1, 1, 1, 1, 1, 0],
                    [0, 1, -1, 2, -2, 0],
                    [0, 1, 1, 4, 4, 0],
                    [0, 1, -1, 8, -8, 1],
                ],
                dtype=self.dtype,
            ),
        )

    def register_winograd_function_2x2_5x5(self) -> None:
        """Register 2x2 output from 5x5 input winograd function"""
        m, r = 2, 5
        self.register_winograd_function(
            m,
            r,
            g=np.array(
                [
                    [1.0 / 4.0, 0, 0, 0, 0],
                    [-1.0 / 6.0, -1.0 / 6.0, -1.0 / 6.0, -1.0 / 6.0, -1.0 / 6.0],
                    [-1.0 / 6.0, 1.0 / 6.0, -1.0 / 6.0, 1.0 / 6.0, -1.0 / 6.0],
                    [1.0 / 24.0, 1.0 / 12.0, 1.0 / 6.0, 1.0 / 3.0, 2.0 / 3.0],
                    [1.0 / 24.0, -1.0 / 12.0, 1.0 / 6.0, -1.0 / 3.0, 2.0 / 3.0],
                    [0, 0, 0, 0, 1],
                ],
                dtype=self.dtype,
            ),
            bt=np.array(
                [
                    [4, 0, -5, 0, 1, 0],
                    [0, -4, -4, 1, 1, 0],
                    [0, 4, -4, -1, 1, 0],
                    [0, -2, -1, 2, 1, 0],
                    [0, 2, -1, -2, 1, 0],
                    [0, 4, 0, -5, 0, 1],
                ],
                dtype=self.dtype,
            ),
            at=np.array([[1, 1, 1, 1, 1, 0], [0, 1, -1, 2, -2, 1]], dtype=self.dtype),
        )

    def __init__(
        self,
        kh: int,
        kw: int,
        vstride: int,
        hstride: int,
        vdilation: int,
        hdilation: int,
        dtype: np.dtype = np.dtype(np.float32),
        tensor_format: TensorFormat = TensorFormat.NCHW,
        debug: bool = False,
        parent_layer: Layerable | None = None,
    ) -> None:
        """
        Initializes the ConvWinograd layer.

        Loading the necessary library and registering available Winograd routines.

        Parameters
        ----------
        kh: int
            Kernel height.
        kw: int
            Kernel width.
        vstride: int
            Vertical stride.
        hstride: int
            Horizontal stride.
        vdilation: int
            Vertical dilation.
        hdilation: int
            Horizontal dilation.
        dtype: np.dtype, optional
            The element data type being used on all the matrices. Defaults to np.float32.
        tensor_format: TensorFormat, optional
            The format of the input and output tensors (NCHW or NHWC). Defaults to TensorFormat.NCHW.
        debug: bool, optional
            Whether to print debug information or not. Defaults to False.
        parent_layer: object, optional
            The layer that is using this Winograd implementation (for tracing purposes). Defaults to None.
        """

        # Parent layer
        if parent_layer is not None:
            self.get_parent_layer = weakref.ref(parent_layer)
            self.evaluate_only = parent_layer.model.evaluate_only
            # enable_best_of = self.get_parent_layer().model.enable_best_of
        else:
            self.evaluate_only = True
        enable_best_of = True

        if isinstance(dtype, np.dtype):
            self.dtype = dtype
        else:
            try:
                self.dtype = {"float32": np.float32, "float64": np.float64}[dtype]
            except KeyError:
                raise NotImplementedError("dtype '{}' not recognized".format(dtype))

        self.tensor_format = tensor_format

        if ConvWinograd.lib_cw is None:
            ConvWinograd.lib_cw = load_library("convwinograd")

        self.alternatives = defaultdict(lambda: [])
        m, r = None, None

        if (kh, kw) == (2, 2) and (vstride, hstride) == (1, 1) and (vdilation, hdilation) == (1, 1):
            self.register_winograd_function_3x3_2x2()
            m, r = 3, 2

        if (kh, kw) == (3, 3) and (vstride, hstride) == (1, 1) and (vdilation, hdilation) == (1, 1):
            self.register_winograd_function_2x2_3x3()
            m, r = 2, 3

        if (kh, kw) == (3, 3) and (vstride, hstride) == (1, 1) and (vdilation, hdilation) == (1, 1):
            self.register_winograd_function_4x4_3x3()
            m, r = 4, 3

        if (kh, kw) == (5, 5) and (vstride, hstride) == (1, 1) and (vdilation, hdilation) == (1, 1):
            self.register_winograd_function_2x2_5x5()
            m, r = 2, 5

        m  # pyright: ignore[reportUnusedExpression] (fake use of m)

        if r not in self.alternatives:
            raise NotImplementedError(f"Winograd not implemented for kernel {kh}x{kw}")

        try:
            self.conv_winograd_workspace_alloc_pre = getattr(
                self.__class__.lib_cw, "conv_winograd_workspace_alloc_pre"
            )
            self.conv_winograd_workspace_alloc_kernel = getattr(
                self.__class__.lib_cw, "conv_winograd_workspace_alloc_kernel"
            )
        except AttributeError:
            logger.error("Winograd conv_winograd_workspace_alloc_pre/kernel routines not found.")

        self.cw_cache_pre = lambda args: self.winograd_workspace_alloc_pre(*args)  # MemoryCache
        self.cw_cache_kernel = lambda args: self.winograd_workspace_alloc_kernel(
            *args
        )  # MemoryCache
        self.y_cache = lambda shape: np.zeros(shape, self.dtype)  # MemoryCache
        self.d_cache = lambda shape: np.zeros(shape, self.dtype)  # MemoryCache

        # Debug
        self.debug = debug

        self._reuse_processed_weights = False
        if self.evaluate_only:
            self._reuse_processed_weights = True
        self._weights_already_processed = False

        if enable_best_of and len(self.alternatives[r]) > 1:
            setattr(
                self,
                f"conv_winograd_{self.tensor_format}",
                BestOf(
                    name="Winograd functions",
                    alternatives=self.alternatives[r],
                    get_problem_size=lambda *args, **kwargs: tuple(
                        list(args[0].shape) + list(args[1].shape[1:])
                    ),
                ),
            )
        else:
            setattr(self, f"conv_winograd_{self.tensor_format}", self.alternatives[r][0][1])

    def conv_winograd_workspace_alloc_kernel(
        self,
        m: ctypes.c_uint,
        r: ctypes.c_uint,
        n: ctypes.c_uint,
        k: ctypes.c_uint,
        c: ctypes.c_uint,
        hi: ctypes.c_uint,
        wi: ctypes.c_uint,
        kh: ctypes.c_uint,
        kw: ctypes.c_uint,
        vpadding: ctypes.c_uint,
        hpadding: ctypes.c_uint,
        _v: ctypes.c_void_p,
        _m: ctypes.c_void_p,
    ) -> Any:
        """
        Placeholder for the C function to allocate kernel workspace memory.

        This method is intended to be overridden or called by the C library.
        """
        pass

    def conv_winograd_workspace_alloc_pre(
        self,
        m: ctypes.c_uint,
        r: ctypes.c_uint,
        k: ctypes.c_uint,
        c: ctypes.c_uint,
        _u: ctypes.c_void_p,
    ) -> Any:
        """
        Placeholder for the C function to allocate pre-processing workspace memory.

        This method is intended to be overridden or called by the C library.
        """
        pass

    def conv_winograd_nchw(
        self,
        weights: np.ndarray,
        x: np.ndarray,
        biases: np.ndarray | None,
        vpadding: int,
        hpadding: int,
        vstride: int,
        hstride: int,
        vdilation: int,
        hdilation: int,
    ) -> np.ndarray:
        """
        Abstract method to perform Winograd convolution in NCHW format.

        This method should be implemented by specific Winograd routines or
        delegated to the chosen optimized function.

        Parameters
        ----------
        weights: np.ndarray
            The convolution weights.
        x: np.ndarray
            The input tensor.
        biases: np.ndarray or None
            The bias tensor.
        vpadding: int
            Vertical padding.
        hpadding: int
            Horizontal padding.
        vstride: int
            Vertical stride.
        hstride: int
            Horizontal stride.
        vdilation: int
            Vertical dilation.
        hdilation: int
            Horizontal dilation.

        Returns
        -------
        np.ndarray
            The output tensor after convolution.

        Raises
        ------
        NotImplementedError
            If the method is called directly.
        """
        raise NotImplementedError("Abstract method called!")

    def conv_winograd_nhwc(
        self,
        weights: np.ndarray,
        x: np.ndarray,
        biases: np.ndarray | None,
        vpadding: int,
        hpadding: int,
        vstride: int,
        hstride: int,
        vdilation: int,
        hdilation: int,
    ) -> np.ndarray:
        """
        Abstract method to perform Winograd convolution in NHWC format.

        This method should be implemented by specific Winograd routines or
        delegated to the chosen optimized function.

        Parameters
        ----------
        weights: np.ndarray
            The convolution weights.
        x: np.ndarray
            The input tensor.
        biases: np.ndarray or None
            The bias tensor.
        vpadding: int
            Vertical padding.
        hpadding: int
            Horizontal padding.
        vstride: int
            Vertical stride.
        hstride: int
            Horizontal stride.
        vdilation: int
            Vertical dilation.
        hdilation: int
            Horizontal dilation.

        Returns
        -------
        np.ndarray
            The output tensor after convolution.

        Raises
        ------
        NotImplementedError
            If the method is called directly.
        """
        raise NotImplementedError("Abstract method called!")

    def encode_shape(self, shape: tuple) -> tuple:
        """
        Encodes a shape tuple according to the configured tensor format.

        Parameters
        ----------
        shape: tuple
            The shape tuple to encode.

        Returns
        -------
        tuple
            The encoded shape tuple.
        """
        return encode_shape(shape, self.tensor_format)

    def decode_shape(self, shape: tuple) -> tuple:
        """
        Decodes a shape tuple according to the configured tensor format.

        Parameters
        ----------
        shape: tuple
            The shape tuple to decode.

        Returns
        -------
        tuple
            The decoded shape tuple.
        """
        return decode_shape(shape, self.tensor_format)

    def _conv_winograd_numpy(  # noqa: C901
        self,
        m: int,
        r: int,
        g: np.ndarray,
        bt: np.ndarray,
        at: np.ndarray,
        pre: Callable,
        kernel: Callable,
        weights: np.ndarray,
        x: np.ndarray,
        biases: np.ndarray | None = None,
        vpadding: int = 0,
        hpadding: int = 0,
        vstride: int = 1,
        hstride: int = 1,
        vdilation: int = 1,
        hdilation: int = 1,
        relu: bool = False,
        bn: bool = False,
        running_mean: np.ndarray | None = None,
        inv_std: np.ndarray | None = None,
        gamma: np.ndarray | None = None,
        beta: np.ndarray | None = None,
    ) -> np.ndarray:
        """
        Performs Winograd convolution using a NumPy-based implementation.

        This method serves as a fallback when optimized C/C++ routines are not
        available or selected. It implements the Winograd algorithm steps using
        NumPy operations.

        Parameters
        ----------
        m: int
            Winograd transform parameter 'm' (output tile size).
        r: int
            Winograd transform parameter 'r' (input tile size).
        g: np.ndarray
            Transformation matrix G for input data.
        bt: np.ndarray
            Transformation matrix B_T for input data.
        at: np.ndarray
            Transformation matrix A_T for output data.
        pre: callable
            Function for pre-processing (e.g., workspace allocation).
        kernel: callable
            Function for kernel execution (e.g., workspace allocation).
        weights: np.ndarray
            The convolution weights (shape depends on tensor_format).
        x: np.ndarray
            The input tensor (shape depends on tensor_format).
        biases: np.ndarray or None, optional
            The bias tensor. Defaults to None.
        vpadding: int, optional
            Vertical padding. Defaults to 0.
        hpadding: int, optional
            Horizontal padding. Defaults to 0.
        vstride: int, optional
            Vertical stride. Defaults to 1.
        hstride: int, optional
            Horizontal stride. Defaults to 1.
        vdilation: int, optional
            Vertical dilation. Defaults to 1.
        hdilation: int, optional
            Horizontal dilation. Defaults to 1.
        relu: bool, optional
            Whether to apply ReLU activation. Defaults to False.
        bn: bool, optional
            Whether to apply Batch Normalization. Defaults to False.
        running_mean: np.ndarray or None, optional
            Running mean for Batch Normalization. Defaults to None.
        inv_std: np.ndarray or None, optional
            Inverse standard deviation for Batch Normalization. Defaults to None.
        gamma: np.ndarray or None, optional
            Scale parameter for Batch Normalization. Defaults to None.
        beta: np.ndarray or None, optional
            Shift parameter for Batch Normalization. Defaults to None.

        Returns
        -------
        np.ndarray
            The output tensor after convolution.

        Raises
        ------
        ValueError
            If kernel size, stride, or dilation are not supported by this Winograd version.
        NotImplementedError
            If the tensor format is not supported.
        """

        n, ci, hi, wi = self.decode_shape(x.shape)

        if self.tensor_format == TensorFormat.NCHW:
            co, ci, kh, kw = weights.shape
        else:
            ci, kh, kw, co = weights.shape

        t = m + r - 1  # Winograd sliding window size t x t
        s = m  # Winograd sliding window stride: t - (r - 1) = m

        if (kh, kw) != (r, r):
            raise ValueError(
                f"Kernel size {kh}x{kw} not supported by this version of Winograd, kernel size"
                f" should be ({r}x{r})!"
            )

        if (vstride, hstride) != (1, 1):
            raise ValueError(
                f"Stride {vstride}x{hstride} supported by this version of Winograd, stride should"
                " be (1,1)!"
            )

        if (vdilation, hdilation) != (1, 1):
            raise ValueError(
                f"Dilation {vdilation}x{hdilation} supported by this version of Winograd, dilation"
                " should be (1,1)!"
            )

        ho = (hi + 2 * vpadding - vdilation * (kh - 1) - 1) // vstride + 1
        wo = (wi + 2 * hpadding - hdilation * (kw - 1) - 1) // hstride + 1

        tile_h = math.ceil((hi + 2 * vpadding - t) / s) + 1
        tile_w = math.ceil((wi + 2 * hpadding - t) / s) + 1

        y_shape = self.encode_shape((n, co, ho, wo))
        y = self.y_cache(y_shape)
        u = np.zeros(
            (t, t, co, ci), self.dtype
        )  # FIXME: self.u_cache[(t, t, co, ci)]  # Workspace for G * g * G^T
        v = np.zeros(
            (t, t, ci, (n * tile_h * tile_w)), self.dtype
        )  # FIXME: self.v_cache[(t, t, ci, (n * tile_h * tile_w))]
        # m_= self.m_cache[(t, t, co, (n * tile_h * tile_w))]
        d = self.d_cache((t, t))

        for k in range(co):
            for c in range(ci):
                # U = G  * g * G^T
                u[..., k, c] = (g @ weights[k, c, ...]) @ g.T

        # 1.1) First alternative: padding first
        # x_padded = best_pad(x, vpadding, hpadding)
        # _, _, hi, wi = x_padded.shape

        # for c in range(ci):
        #     for b in range(n):
        #         for h in range(tile_h):
        #             for w in range(tile_w):
        #                 hh, ww = h * s, w * s
        #                 th, tw = min(hi-hh,t), min(wi-ww,t)
        #                 d[:th,:tw] = x_padded[b, c, hh:hh+th, ww:ww+tw]
        #                 v[..., c, b * tile_h * tile_w + h * tile_w + w] = (self.bt @ d) @ self.bt.T

        # 1.2) Second alternative: avoid padding
        for c in range(ci):
            for b in range(n):
                for h in range(tile_h):
                    hh_ = min(hi, h * s - vpadding)
                    hh, fh = (hh_, 0) if hh_ > 0 else (0, min(-hh_, t))
                    oh = max(min(t, hi - hh) - min(t, fh), 0)

                    for w in range(tile_w):
                        ww_ = min(wi, w * s - hpadding)
                        ww, fw = (ww_, 0) if ww_ > 0 else (0, min(-ww_, t))
                        ow = max(min(t, wi - ww) - min(t, fw), 0)

                        if hh < hh + oh and ww < ww + ow:
                            match self.tensor_format:
                                case TensorFormat.NCHW:
                                    d[fh: fh + oh, fw: fw + ow] = x[
                                        b, c, hh: hh + oh, ww: ww + ow
                                    ]
                                case TensorFormat.NHWC:
                                    d[fh: fh + oh, fw: fw + ow] = x[
                                        b, hh: hh + oh, ww: ww + ow, c
                                    ]
                                case tensor_format:
                                    raise NotImplementedError(
                                        f"Unsupported tensor format {tensor_format}!"
                                    )

                        #   0  0  0
                        #   X  X  X
                        #   X  X  X
                        if 0 <= fh:
                            d[:fh, ...] = 0

                        #   0  0  0
                        #   X  X  X
                        #   0  0  0
                        if fh + oh < t:
                            d[fh + oh:, ...] = 0

                        #   0  0  0
                        #   0  X  X
                        #   0  0  0
                        if 0 <= fw:
                            d[fh: fh + oh, :fw] = 0

                        #   0  0  0
                        #   0  X  0
                        #   0  0  0
                        if fw + ow < t:
                            d[fh: fh + oh, fw + ow:] = 0

                        v[..., c, b * tile_h * tile_w + h * tile_w + w] = (bt @ d) @ bt.T

        # 2.1) First alternative: np.einsum
        m_ = np.einsum("... i j, ... j k -> ... i k", u, v)

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
                        match self.tensor_format:
                            case TensorFormat.NCHW:
                                y[b, k, hh: hh + m, ww: ww + m] = z[
                                    : min(m, ho - hh), : min(m, wo - ww)
                                ]
                            case TensorFormat.NHWC:
                                y[b, hh: hh + m, ww: ww + m, k] = z[
                                    : min(m, ho - hh), : min(m, wo - ww)
                                ]
                            case tensor_format:
                                raise NotImplementedError(
                                    f"Unsupported tensor format {tensor_format}!"
                                )

            if biases is not None:
                match self.tensor_format:
                    case TensorFormat.NCHW:
                        y[:, k, ...] += biases[k]
                    case TensorFormat.NHWC:
                        y[..., k] += biases[k]
                    case tensor_format:
                        raise NotImplementedError(f"Unsupported tensor format {tensor_format}!")

            if bn:
                assert running_mean
                assert inv_std
                assert gamma
                assert beta
                match self.tensor_format:
                    case TensorFormat.NCHW:
                        y[:, k, ...] = (
                            ((y[:, k, ...] - running_mean[k]) * inv_std[k]) * gamma[k]
                        ) + beta[k]
                    case TensorFormat.NHWC:
                        y[..., k] = (
                            ((y[..., k] - running_mean[k]) * inv_std[k]) * gamma[k]
                        ) + beta[k]
                    case tensor_format:
                        raise NotImplementedError(f"Unsupported tensor format {tensor_format}!")

        if relu:
            y[y < 0] = 0

        return y

    def _conv_winograd_c(
        self,
        m: int,
        r: int,
        g: np.ndarray,
        bt: np.ndarray,
        at: np.ndarray,
        x_winograd_pre: Callable,
        x_winograd_kernel: Callable,
        weights: np.ndarray,
        x: np.ndarray,
        biases: np.ndarray | None = None,
        vpadding: int = 0,
        hpadding: int = 0,
        vstride: int = 1,
        hstride: int = 1,
        vdilation: int = 1,
        hdilation: int = 1,
        relu: bool = False,
        bn: bool = False,
        running_mean: np.ndarray | None = None,
        inv_std: np.ndarray | None = None,
        gamma: np.ndarray | None = None,
        beta: np.ndarray | None = None,
    ) -> np.ndarray:
        """
        Performs Winograd convolution using an optimized C/C++ library routine.

        This method orchestrates the call to the underlying compiled library functions
        for Winograd convolution, including workspace allocation and kernel execution.
        It handles data type conversions and parameter passing to the C functions.

        Parameters
        ----------
        m: int
            Winograd transform parameter 'm' (output tile size).
        r: int
            Winograd transform parameter 'r' (input tile size).
        g: np.ndarray
            Transformation matrix G for input data.
        bt: np.ndarray
            Transformation matrix B_T for input data.
        at: np.ndarray
            Transformation matrix A_T for output data.
        x_winograd_pre: callable
            The C function pointer for Winograd pre-processing (e.g., weight transformation).
        x_winograd_kernel: callable
            The C function pointer for Winograd kernel execution.
        weights: np.ndarray
            The convolution weights (shape depends on tensor_format).
        x: np.ndarray
            The input tensor (shape depends on tensor_format).
        biases: np.ndarray or None, optional
            The bias tensor. Defaults to None.
        vpadding: int, optional
            Vertical padding. Defaults to 0.
        hpadding: int, optional
            Horizontal padding. Defaults to 0.
        vstride: int, optional
            Vertical stride. Defaults to 1.
        hstride: int, optional
            Horizontal stride. Defaults to 1.
        vdilation: int, optional
            Vertical dilation. Defaults to 1.
        hdilation: int, optional
            Horizontal dilation. Defaults to 1.
        relu: bool, optional
            Whether to apply ReLU activation. Defaults to False.
        bn: bool, optional
            Whether to apply Batch Normalization. Defaults to False.
        running_mean: np.ndarray or None, optional
            Running mean for Batch Normalization. Defaults to None.
        inv_std: np.ndarray or None, optional
            Inverse standard deviation for Batch Normalization. Defaults to None.
        gamma: np.ndarray or None, optional
            Scale parameter for Batch Normalization. Defaults to None.
        beta: np.ndarray or None, optional
            Shift parameter for Batch Normalization. Defaults to None.

        Returns
        -------
        np.ndarray
            The output tensor after convolution.

        Raises
        ------
        ValueError
            If kernel size, stride, or dilation are not supported by this Winograd version.
        NotImplementedError
            If the tensor format is not supported.
        """

        n, ci, hi, wi = self.decode_shape(x.shape)

        match self.tensor_format:
            case TensorFormat.NCHW:
                co, ci, kh, kw = weights.shape
            case TensorFormat.NHWC:
                ci, kh, kw, co = weights.shape
            case tensor_format:
                raise NotImplementedError(f"Unsupported tensor format {tensor_format}!")

        # t = m + r - 1  # Winograd sliding window size t x t
        # s = m  # Winograd sliding window stride: t - (r - 1) = m

        if (kh, kw) != (r, r):
            raise ValueError(
                f"Kernel size {kh}x{kw} not supported by this version of Winograd, kernel size"
                f" should be ({r}x{r})!"
            )

        if (vstride, hstride) != (1, 1):
            raise ValueError(
                f"Stride {vstride}x{hstride} supported by this version of Winograd, stride should"
                " be (1,1)!"
            )

        if (vdilation, hdilation) != (1, 1):
            raise ValueError(
                f"Dilation {vdilation}x{hdilation} supported by this version of Winograd, dilation"
                " should be (1,1)!"
            )

        ho = (hi + 2 * vpadding - vdilation * (kh - 1) - 1) // vstride + 1
        wo = (wi + 2 * hpadding - hdilation * (kw - 1) - 1) // hstride + 1

        (
            _weights_already_processed,
            _u,
        ) = self.cw_cache_pre((m, r, co, ci))
        _v, _m = self.cw_cache_kernel((m, r, n, co, ci, hi, wi, kh, kw, vpadding, hpadding))

        y_shape = self.encode_shape((n, co, ho, wo))
        y = self.y_cache(y_shape)

        match self.tensor_format:
            case TensorFormat.NCHW:
                ld_d1, ld_d2, ld_d3 = ci * hi * wi, hi * wi, wi
                ld_f1, ld_f2, ld_f3 = ci * kh * kw, kh * kw, kw
                ld_y1, ld_y2, ld_y3 = co * ho * wo, ho * wo, wo
            case TensorFormat.NHWC:
                ld_d1, ld_d2, ld_d3 = hi * wi * ci, wi * ci, ci
                ld_f1, ld_f2, ld_f3 = kh * kw * co, kw * co, co
                ld_y1, ld_y2, ld_y3 = ho * wo * co, wo * co, co
            case tensor_format:
                raise NotImplementedError(f"Unsupported tensor format {tensor_format}!")

        if not self._reuse_processed_weights or not _weights_already_processed[0]:
            x_winograd_pre(
                ctypes.c_uint(m),
                ctypes.c_uint(r),
                ctypes.c_uint(n),
                ctypes.c_uint(co),
                ctypes.c_uint(ci),
                ctypes.c_uint(kh),
                ctypes.c_uint(kw),
                ctypes.c_void_p(weights.ctypes.data),
                ctypes.c_uint(ld_f1),
                ctypes.c_uint(ld_f2),
                ctypes.c_uint(ld_f3),
                _u,
            )
            _weights_already_processed[0] = True

        x_winograd_kernel(
            ctypes.c_uint(m),
            ctypes.c_uint(r),
            ctypes.c_uint(n),
            ctypes.c_uint(co),
            ctypes.c_uint(ci),
            ctypes.c_uint(hi),
            ctypes.c_uint(wi),
            ctypes.c_uint(kh),
            ctypes.c_uint(kw),
            ctypes.c_uint(vpadding),
            ctypes.c_uint(hpadding),
            ctypes.c_void_p(x.ctypes.data),
            ctypes.c_uint(ld_d1),
            ctypes.c_uint(ld_d2),
            ctypes.c_uint(ld_d3),
            ctypes.c_void_p(y.ctypes.data),
            ctypes.c_uint(ld_y1),
            ctypes.c_uint(ld_y2),
            ctypes.c_uint(ld_y3),
            ctypes.c_void_p(None if biases is None else biases.ctypes.data),
            _u,
            _v,
            _m,
            ctypes.c_char((b"F", b"T")[relu]),
            ctypes.c_char((b"F", b"T")[bn]),
            ctypes.c_void_p(None if running_mean is None else running_mean.ctypes.data),
            ctypes.c_void_p(None if inv_std is None else inv_std.ctypes.data),
            ctypes.c_void_p(None if gamma is None else gamma.ctypes.data),
            ctypes.c_void_p(None if beta is None else beta.ctypes.data),
        )
        return y


def time_it_func(
    x: np.ndarray,
    w_c: np.ndarray,
    biases: np.ndarray,
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
) -> None:
    """
    Times the execution of a convolution operation using im2row (NHWC format) and matrix multiplication.

    This function is primarily for benchmarking and comparison purposes. It reshapes
    the input data using im2row and then performs a matrix multiplication with weights.

    Parameters
    ----------
    x: np.ndarray
        Input tensor in NHWC format.
    w_c: np.ndarray
        Weights reshaped for matrix multiplication.
    biases: np.ndarray
        Bias vector.
    b: int
        Batch size.
    kn: int
        Number of output channels (filters).
    ho: int
        Output height.
    wo: int
        Output width.
    kh: int
        Kernel height.
    kw: int
        Kernel width.
    vpadding: int
        Vertical padding.
    hpadding: int
        Horizontal padding.
    vstride: int
        Vertical stride.
    hstride: int
        Horizontal stride.
    vdilation: int
        Vertical dilation.
    hdilation: int
        Horizontal dilation.
    """

    res = np.zeros(((x.shape[0] * ho * wo), (x.shape[-1] * kh * kw)), dtype=x.dtype)
    im2row_nhwc_cython(
        x,
        res,
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
    res += biases.reshape(b * ho * wo, kn)


def time_it_im2col(
    x: np.ndarray,
    w_c: np.ndarray,
    biases: np.ndarray,
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
) -> None:
    """
    Times the execution of a convolution operation using im2col (NCHW format) and matrix multiplication.

    This function is primarily for benchmarking and comparison purposes. It reshapes
    the input data using im2col and then performs a matrix multiplication with weights.

    Parameters
    ----------
    x: np.ndarray
        Input tensor in NCHW format.
    w_c: np.ndarray
        Weights reshaped for matrix multiplication.
    biases: np.ndarray
        Bias vector.
    b: int
        Batch size.
    kn: int
        Number of output channels (filters).
    ho: int
        Output height.
    wo: int
        Output width.
    kh: int
        Kernel height.
    kw: int
        Kernel width.
    vpadding: int
        Vertical padding.
    hpadding: int
        Horizontal padding.
    vstride: int
        Vertical stride.
    hstride: int
        Horizontal stride.
    vdilation: int
        Vertical dilation.
    hdilation: int
        Horizontal dilation.
    """

    res = np.zeros(((x.shape[0] * ho * wo), (x.shape[-1] * kh * kw)), dtype=x.dtype)
    im2col_nchw_cython(
        x,
        res,
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
    res += biases.reshape(b * ho * wo, kn)


def time_it_im2col_4_dims(
    x: np.ndarray,
    w_c: np.ndarray,
    biases: np.ndarray,
    kk: int,
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
) -> None:
    """
    Times the execution of a convolution operation using im2col (NCHW format) and matrix multiplication.

    (specifically for a 4-dimensional output shape)

    This function is primarily for benchmarking and comparison purposes. It reshapes
    the input data using im2col and then performs a matrix multiplication with weights,
    adjusting the output shape for 4 dimensions.

    Parameters
    ----------
    x: np.ndarray
        Input tensor in NCHW format.
    w_c: np.ndarray
        Weights reshaped for matrix multiplication.
    biases: np.ndarray
        Bias vector.
    kk: int
        Number of output channels (filters).
    ho: int
        Output height.
    wo: int
        Output width.
    kh: int
        Kernel height.
    kw: int
        Kernel width.
    vpadding: int
        Vertical padding.
    hpadding: int
        Horizontal padding.
    vstride: int
        Vertical stride.
    hstride: int
        Horizontal stride.
    vdilation: int
        Vertical dilation.
    hdilation: int
        Horizontal dilation.
    """

    res = np.zeros(((x.shape[0] * ho * wo), (x.shape[-1] * kh * kw)), dtype=x.dtype)
    im2col_nchw_cython(
        x,
        res,
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
    res += biases.reshape(kk, -1, ho, wo).transpose(1, 0, 2, 3)


def main() -> None:
    """
    Provides a usage example for the ConvWinograd class.

    This function demonstrates how to instantiate and use the ConvWinograd class
    for both NCHW and NHWC tensor formats. It compares the results and performance
    against a standard im2col + matrix multiplication approach. It also includes
    a more extensive test loop to check various configurations.

    The example requires `timeit` and `pydtnn.utils.random` to be imported.
    """
    # Imports for this usage example (not required otherwise)
    from timeit import timeit

    from pydtnn.utils import rand

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
    # Create weights, x, and biases matrices from previous parameters. If no biases
    # matrix is provided, a proper one filled with zeros will be automatically
    # created.
    rand.seed(0)
    # weights[1][1][1][1] = -322.0
    # weights[2][2][2][2] = -334.0

    ho = (h + 2 * vpadding - vdilation * (kh - 1) - 1) // vstride + 1
    wo = (w + 2 * hpadding - hdilation * (kw - 1) - 1) // hstride + 1

    biases_wg = (np.ones(kn) * 10).astype(np.float32, order="C")

    # NCHW --------------------------
    weights = rand.random((kn, c, kh, kw)).astype(np.float32, order="C")
    x = rand.random((b, c, h, w)).astype(np.float32, order="C")
    biases = (np.ones((kn, b * ho * wo)) * 10).astype(np.float32, order="C")
    logger.info("Using conv_winograd NCHW to compute weights * x + biases...")
    conv_winograd = ConvWinograd(kh, kw, vstride, hstride, vdilation, hdilation, debug=False)
    conv_winograd_result_nchw = conv_winograd.conv_winograd_nchw(
        weights,
        x,
        biases_wg,
        vpadding=vpadding,
        hpadding=hpadding,
        vstride=vstride,
        hstride=hstride,
        vdilation=vdilation,
        hdilation=hdilation,
    )
    conv_winograd_t = (
        timeit(
            lambda: conv_winograd.conv_winograd_nchw(
                weights,
                x,
                biases_wg,
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
    logger.info("conv_winograd time: {:.4f}".format(conv_winograd_t))
    logger.info("Using im2col and mm NCHW ...")
    x_c = np.zeros((c * kh * kw, b * ho * wo))
    im2col_nchw_cython(
        x, x_c, kh, kw, ho, wo, vpadding, hpadding, vstride, hstride, vdilation, hdilation
    )
    w_c = weights.reshape(kn, -1)
    im2col_mm_result_nchw = (w_c @ x_c + biases).reshape(kn, -1, ho, wo).transpose(1, 0, 2, 3)
    mm_t = (
        timeit(
            lambda: time_it_im2col(
                x,
                w_c,
                biases,
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
    logger.info("mm time: {:.4f}".format(mm_t))

    # NHWC --------------------------
    weights = rand.random((c, kh, kw, kn)).astype(np.float32, order="C")
    x = rand.random((b, h, w, c)).astype(np.float32, order="C")
    biases = (np.ones((b * ho * wo, kn)) * 10).astype(np.float32, order="C")
    logger.info("Using conv_winograd NHWC to compute weights * x + biases...")
    conv_winograd = ConvWinograd(
        kh, kw, vstride, hstride, vdilation, hdilation, tensor_format=TensorFormat.NHWC, debug=False
    )
    conv_winograd_result_nhwc = conv_winograd.conv_winograd_nhwc(
        weights,
        x,
        biases_wg,
        vpadding=vpadding,
        hpadding=hpadding,
        vstride=vstride,
        hstride=hstride,
        vdilation=vdilation,
        hdilation=hdilation,
    )
    conv_winograd_t = (
        timeit(
            lambda: conv_winograd.conv_winograd_nhwc(
                weights,
                x,
                biases_wg,
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
    logger.info("conv_winograd time: {:.4f}".format(conv_winograd_t))
    logger.info("Using im2col and mm NHWC ...")
    x_c = np.zeros(((x.shape[0] * ho * wo), (x.shape[-1] * kh * kw)), dtype=x.dtype)
    im2row_nhwc_cython(
        x, x_c, kh, kw, ho, wo, vpadding, hpadding, vstride, hstride, vdilation, hdilation
    )
    w_c = weights.reshape((-1, kn), copy=False)
    im2col_mm_result_nhwc = (x_c @ w_c + biases).reshape(-1, ho, wo, kn)
    mm_t = (
        timeit(
            lambda: time_it_func(
                x,
                w_c,
                biases,
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
    logger.info("mm time: {:.4f}".format(mm_t))

    logger.info(
        "\n".join(
            [
                f"Sum WINOGRAD NCHW: {conv_winograd_result_nchw.sum()} {
                    conv_winograd_result_nchw.shape
                }",
                f"Sum   IM2COL NCHW: {im2col_mm_result_nchw.sum()} {im2col_mm_result_nchw.shape}",
                f"np.allclose NCHW: {
                    np.allclose(conv_winograd_result_nchw, im2col_mm_result_nchw, atol=1e-3)
                }",
                "",
                f"Sum WINOGRAD NHWC: {conv_winograd_result_nhwc.sum()} {
                    conv_winograd_result_nhwc.shape
                }",
                f"Sum   IM2COL NHWC: {im2col_mm_result_nhwc.sum()} {im2col_mm_result_nhwc.shape}",
                f"np.allclose NHWC: {
                    np.allclose(conv_winograd_result_nhwc, im2col_mm_result_nhwc, atol=1e-3)
                }",
            ]
        )
    )
    # """

    # n = 65
    # c = k = 65
    # h = w = 33
    # vpadd = hpadd = 6
    n = 17
    c = k = 17
    h = w = 33
    vpadd = hpadd = 6
    for nn in range(16, n, 16):
        for cc in range(16, c, 16):
            for kk in range(16, k, 16):
                for hh in range(8, h, 4):
                    for vpadding in range(1, vpadd):
                        for hpadding in range(1, hpadd):
                            for kh in [2, 3, 5]:
                                kw = kh
                                ww = hh
                                ho = (hh + 2 * vpadding - vdilation * (kh - 1) - 1) // vstride + 1
                                wo = (ww + 2 * hpadding - hdilation * (kw - 1) - 1) // hstride + 1

                                for tensor_fmt in [TensorFormat.NCHW, TensorFormat.NHWC]:
                                    conv_winograd = ConvWinograd(
                                        kh,
                                        kw,
                                        vstride,
                                        hstride,
                                        vdilation,
                                        hdilation,
                                        tensor_format=tensor_fmt,
                                        debug=False,
                                    )
                                    logger.info(
                                        f"{nn} {cc} {kk} {hh} {ww} {vpadding} {hpadding} {kh} {
                                            conv_winograd.tensor_format
                                        }"
                                    )

                                    biases_wg = (np.ones(kk) * 10).astype(np.float32, order="C")
                                    match tensor_fmt:
                                        case TensorFormat.NCHW:
                                            weights = rand.random((kk, cc, kh, kw)).astype(
                                                np.float32, order="C"
                                            )
                                            x = rand.random((nn, cc, hh, ww)).astype(
                                                np.float32, order="C"
                                            )
                                            biases = (np.ones((kk, nn * ho * wo)) * 10).astype(
                                                np.float32, order="C"
                                            )
                                            w_c = weights.reshape(kk, -1)
                                            res = np.zeros(
                                                ((x.shape[0] * ho * wo), (x.shape[-1] * kh * kw)),
                                                dtype=x.dtype,
                                            )
                                            im2col_nchw_cython(
                                                x,
                                                res,
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
                                            im2col_mm_result = (w_c @ res) + biases
                                            im2col_mm_result = im2col_mm_result.reshape(
                                                kk, -1, ho, wo
                                            ).transpose(1, 0, 2, 3)
                                            im2col_t = (
                                                timeit(
                                                    lambda: time_it_im2col_4_dims(
                                                        x,
                                                        w_c,
                                                        biases,
                                                        kk,
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

                                            conv_winograd_result = conv_winograd.conv_winograd_nchw(
                                                weights,
                                                x,
                                                biases_wg,
                                                vpadding=vpadding,
                                                hpadding=hpadding,
                                                vstride=vstride,
                                                hstride=hstride,
                                                vdilation=vdilation,
                                                hdilation=hdilation,
                                            )
                                            conv_winograd_t = (
                                                timeit(
                                                    lambda: conv_winograd.conv_winograd_nchw(
                                                        weights,
                                                        x,
                                                        biases_wg,
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
                                        case TensorFormat.NHWC:
                                            weights = rand.random((cc, kh, kw, kk)).astype(
                                                np.float32, order="C"
                                            )
                                            x = rand.random((nn, hh, ww, cc)).astype(
                                                np.float32, order="C"
                                            )
                                            biases = (np.ones((nn * ho * wo, kk)) * 10).astype(
                                                np.float32, order="C"
                                            )

                                            w_c = weights.reshape(-1, kk)
                                            im2col_mm_result = np.zeros(
                                                ((x.shape[0] * ho * wo), (x.shape[-1] * kh * kw)),
                                                dtype=x.dtype,
                                            )
                                            (
                                                im2row_nhwc_cython(
                                                    x,
                                                    x_c,
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
                                                @ w_c
                                                + biases
                                            )  # pyright: ignore[reportUnusedExpression]
                                            im2col_mm_result = im2col_mm_result.reshape(
                                                -1, ho, wo, kk
                                            )
                                            im2col_t = (
                                                timeit(
                                                    lambda: time_it_func(
                                                        x,
                                                        w_c,
                                                        biases,
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

                                            conv_winograd_result = conv_winograd.conv_winograd_nhwc(
                                                weights,
                                                x,
                                                biases_wg,
                                                vpadding=vpadding,
                                                hpadding=hpadding,
                                                vstride=vstride,
                                                hstride=hstride,
                                                vdilation=vdilation,
                                                hdilation=hdilation,
                                            )
                                            conv_winograd_t = (
                                                timeit(
                                                    lambda: conv_winograd.conv_winograd_nhwc(
                                                        weights,
                                                        x,
                                                        biases_wg,
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
                                        case tensor_fmt:
                                            raise NotImplementedError(
                                                f"Unsupported tensor format {tensor_fmt}!"
                                            )

                                    logger.info(
                                        " conv_winograd time: {:.4f} ".format(conv_winograd_t)
                                        + "mm time: {:.4f} ".format(im2col_t)
                                        + "np.allclose: {}".format(
                                            np.allclose(
                                                conv_winograd_result, im2col_mm_result, atol=1e-3
                                            )
                                        )
                                    )
                                    # print(" np.sum:", np.max(np.abs(conv_winograd_result-im2col_mm_result)), end="")
                                    logger.info(
                                        (" WINOGR", " IM2COL")[conv_winograd_t > im2col_t],
                                        im2col_t / conv_winograd_t,
                                    )
    # """


if __name__ == "__main__":
    main()
