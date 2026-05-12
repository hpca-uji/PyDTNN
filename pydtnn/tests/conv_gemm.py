"""
Test suite for verifying the correctness of the ConvGemm implementation.
"""

import inspect
import logging

import numpy as np

from pydtnn.backends.cython.utils.im2row_nhwc_cython import im2row_nhwc_cython
from pydtnn.libs.convGemm import ConvGemm
from pydtnn.tests.abstract.common import D, verbose_test
from pydtnn.tests.abstract.conv_common import ConvCommonTestCase
from pydtnn.utils import print_with_header

__all__ = ("ConvGemmTestCase",)

logger = logging.getLogger(__name__)


class ConvGemmTestCase(ConvCommonTestCase):
    """
    Tests that conv_gemm leads to the same results as i2c and mm.
    """

    # NOTE: Delete parent test to prevent re-export and re-testing
    global ConvCommonTestCase
    del ConvCommonTestCase

    @classmethod
    def _compute_both(
        cls, weights: np.ndarray, x: np.ndarray, biases: np.ndarray | None = None, vpadding=0, hpadding=0, vstride=1, hstride=1, vdilation=1, hdilation=1
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Computes convolution results using both ConvGemm and im2row + matrix multiplication for comparison.

        Returns:
            A tuple containing the ConvGemm result and the im2row matrix multiplication result.
        """
        c, kh, kw, kn = weights.shape
        # b, c, h, w = x.shape
        cg_biases = biases.copy() if biases is not None else None
        conv_gemm_result: np.ndarray = cls._compute(
            weights, x, biases=cg_biases, kh=kh, kw=kw, vpadding=vpadding, hpadding=hpadding, vstride=vstride, hstride=hstride, vdilation=vdilation, hdilation=hdilation
        )
        conv_gemm_result: np.ndarray = conv_gemm_result.reshape((-1, kn), copy=False)

        n, h, w, c = x.shape

        ho = (h + 2 * vpadding - vdilation * (kh - 1) - 1) // vstride + 1
        wo = (w + 2 * hpadding - hdilation * (kw - 1) - 1) // hstride + 1

        dim_n = n * ho * wo
        dim_c = c * kh * kw

        x_c: np.ndarray = np.zeros(shape=(dim_n, dim_c), dtype=x.dtype)

        im2row_nhwc_cython(
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
        w_c = weights.reshape((-1, kn), copy=False)
        im2row_mm_result: np.ndarray = np.matmul(x_c, w_c)
        if biases is not None:
            np.add(im2row_mm_result, biases.reshape((-1, kn), copy=False), out=im2row_mm_result, dtype=im2row_mm_result.dtype)
        if verbose_test():
            print_with_header("{} conv_gemm_result".format(inspect.stack()[1][3]), conv_gemm_result)
            logger.info("Shape: ", conv_gemm_result.shape, " Sum: ", conv_gemm_result.sum(), " Min: ", conv_gemm_result.min(), " Max: ", conv_gemm_result.max())
            print_with_header("{} im2row_mm_result".format(inspect.stack()[1][3]), im2row_mm_result)
            logger.info("Shape: ", im2row_mm_result.shape, " Sum: ", im2row_mm_result.sum(), " Min: ", im2row_mm_result.min(), " Max: ", im2row_mm_result.max())
            logger.info("---")
            logger.info("Maximum difference: ", max([abs(x - y) for x, y in zip(conv_gemm_result.flatten(), im2row_mm_result.flatten())]))
            logger.info("---")
        return conv_gemm_result, im2row_mm_result

    @staticmethod
    def _get_config() -> D:
        """
        Returns the configuration dictionary for the test case.
        """
        return D()

    @staticmethod
    def _compute(weights: np.ndarray, x: np.ndarray, biases: np.ndarray | None = None, kh=1, kw=1, vpadding=0, hpadding=0, vstride=1, hstride=1, vdilation=1, hdilation=1):
        """
        Executes the ConvGemm operation for the given inputs.
        """
        return ConvGemm(debug=False).conv_gemm_nhwc(weights, x, None, vpadding, hpadding, vstride, hstride, vdilation, hdilation, biases)
