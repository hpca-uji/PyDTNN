import inspect
import unittest

import numpy as np

from pydtnn.backends.cpu.utils.im2row_nhwc_cython import im2row_nhwc_cython
from pydtnn.libs.libconvwinograd import ConvWinograd
from pydtnn.tests.abstract.common import verbose_test, D
from pydtnn.tests.abstract.conv_common import ConvCommonTestCase
from pydtnn.utils import print_with_header
from pydtnn.utils.tensor import TensorFormat

# if (kh, kw) == (2, 2) and (vstride, hstride) == (1, 1) and (vdilation, hdilation) == (1, 1):
# if (kh, kw) == (3, 3) and (vstride, hstride) == (1, 1) and (vdilation, hdilation) == (1, 1):
# if (kh, kw) == (3, 3) and (vstride, hstride) == (1, 1) and (vdilation, hdilation) == (1, 1):
# if (kh, kw) == (5, 5) and (vstride, hstride) == (1, 1) and (vdilation, hdilation) == (1, 1):


class ConvWinogradTestCase(ConvCommonTestCase):
    """
    Tests that conv_winograd leads to the same results as i2c and mm.
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
        cw_biases = biases.copy() if biases is not None else None
        conv_winograd_result: np.ndarray = cls._compute(weights, x, biases=cw_biases,
                                                        kh=kh, kw=kw,
                                                        vpadding=vpadding, hpadding=hpadding,
                                                        vstride=vstride, hstride=hstride,
                                                        vdilation=vdilation, hdilation=hdilation)
        conv_winograd_result: np.ndarray = conv_winograd_result.reshape((-1, kn), copy=False)

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
            print_with_header("{} conv_winograd_result".format(inspect.stack()[1][3]), conv_winograd_result)
            print("Shape: ", conv_winograd_result.shape,
                  " Sum: ", conv_winograd_result.sum(),
                  " Min: ", conv_winograd_result.min(),
                  " Max: ", conv_winograd_result.max())
            print_with_header("{} im2row_mm_result".format(inspect.stack()[1][3]), im2row_mm_result)
            print("Shape: ", im2row_mm_result.shape,
                  " Sum: ", im2row_mm_result.sum(),
                  " Min: ", im2row_mm_result.min(),
                  " Max: ", im2row_mm_result.max())
            print("---")
            print("Maximum difference: ",
                  max([abs(x - y) for x, y in zip(conv_winograd_result.flatten(), im2row_mm_result.flatten())]))
            print("---")
        return conv_winograd_result, im2row_mm_result

    @staticmethod
    def _compute(weights: np.ndarray, x: np.ndarray, biases: np.ndarray | None = None, kh=1, kw=1, vpadding=0, hpadding=0, vstride=1, hstride=1, vdilation=1, hdilation=1):
        return ConvWinograd(kh, kw, vstride, hstride, vdilation, hdilation, debug=verbose_test(), tensor_format=TensorFormat.NHWC).conv_winograd_nhwc(weights, x, biases, vpadding, hpadding, vstride, hstride, vdilation, hdilation)

    @staticmethod
    def _get_config() -> D:
        return D(h=100, w=100, kh=3, kw=3)

    @unittest.skip("Winograd only supports stride 1x1")
    def test_raise_on_different_strides(self):
        raise NotImplementedError()

    @unittest.skip("Winograd only supports stride 1x1")
    def test_with_different_stride(self):
        raise NotImplementedError()

    @unittest.skip("Winograd only supports stride 1x1")
    def test_with_different_strides(self):
        raise NotImplementedError()

    @unittest.skip("Winograd only supports dilation 1x1")
    def test_with_different_dilation(self):
        raise NotImplementedError()

    @unittest.skip("Winograd only supports a subset of kernel sizes")
    def test_alexnet_layers(self):
        raise NotImplementedError()
