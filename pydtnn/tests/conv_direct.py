import inspect
from unittest import SkipTest

import numpy as np

from pydtnn.libs.libconvdirect import ConvDirect
from pydtnn.backends.cpu.utils.im2row_nhwc_cython import im2row_nhwc_cython
from pydtnn.tests.abstract.common import verbose_test
from pydtnn.tests.abstract.conv_common import ConvCommonTestCase
from pydtnn.utils import print_with_header


class ConvDirectTestCase(ConvCommonTestCase):
    """
    Tests that conv_direct leads to the same results as i2c and mm.
    """
    # NOTE: Delete parent test to prevent re-export and re-testing
    global ConvCommonTestCase
    del ConvCommonTestCase

    @classmethod
    def _compute_both(cls, weights: np.ndarray, x: np.ndarray, biases: np.ndarray | None = None,
                      vpadding=0, hpadding=0, vstride=1, hstride=1,
                      vdilation=1, hdilation=1) -> tuple[np.ndarray, np.ndarray]:
        c, kh, kw, kn = weights.shape
        # b, c, h, w = x.shape
        cg_biases = biases.copy() if biases is not None else None
        conv_direct_result: np.ndarray = cls._compute(weights, x, biases=cg_biases,
                                                      kw=kw, kh=kh,
                                                      vpadding=vpadding, hpadding=hpadding,
                                                      vstride=vstride, hstride=hstride,
                                                      vdilation=vdilation, hdilation=hdilation)
        conv_direct_result: np.ndarray = conv_direct_result.reshape((-1, kn), copy=False)

        n, h, w, c = x.shape

        ho = (h + 2 * vpadding - vdilation * (kh - 1) - 1) // vstride + 1
        wo = (w + 2 * hpadding - hdilation * (kw - 1) - 1) // hstride + 1

        dim_n = n * ho * wo
        dim_c = c * kh * kw

        x_c: np.ndarray = np.zeros(shape=(dim_n, dim_c), dtype=x.dtype)

        im2row_nhwc_cython(x, x_c,
                           kh, kw, ho, wo,
                           vpadding, hpadding,
                           vstride, hstride,
                           vdilation, hdilation)
        w_c = weights.reshape((-1, kn), copy=False)
        im2row_mm_result: np.ndarray = np.matmul(x_c, w_c)
        if biases is not None:
            np.add(im2row_mm_result, biases.reshape((-1, kn), copy=False), out=im2row_mm_result, dtype=im2row_mm_result.dtype)
        if verbose_test():
            print_with_header("{} conv_direct_result".format(inspect.stack()[1][3]), conv_direct_result)
            print("Shape: ", conv_direct_result.shape,
                  " Sum: ", conv_direct_result.sum(),
                  " Min: ", conv_direct_result.min(),
                  " Max: ", conv_direct_result.max())
            print_with_header("{} im2row_mm_result".format(inspect.stack()[1][3]), im2row_mm_result)
            print("Shape: ", im2row_mm_result.shape,
                  " Sum: ", im2row_mm_result.sum(),
                  " Min: ", im2row_mm_result.min(),
                  " Max: ", im2row_mm_result.max())
            print("---")
            print("Maximum difference: ",
                  max([abs(x - y) for x, y in zip(conv_direct_result.flatten(), im2row_mm_result.flatten())]))
            print("---")
        return conv_direct_result, im2row_mm_result

    @staticmethod
    def _compute(weights: np.ndarray, x: np.ndarray, biases: np.ndarray | None = None, kh=1, kw=1, vpadding=0, hpadding=0, vstride=1, hstride=1, vdilation=1, hdilation=1):
        if biases is not None:
            raise SkipTest("Direct does not support biases")
        return ConvDirect(method_name="convdirect_original_nhwc_default", debug=False).conv_direct(weights, x, None, vpadding, hpadding, vstride, hstride, vdilation, hdilation)
