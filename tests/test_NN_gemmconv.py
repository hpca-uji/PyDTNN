import unittest
import numpy as np
import sys

try:
    from NN_gemmconv import GemmConv
    from NN_im2col_cython import im2col_cython
except ModuleNotFoundError:
    print("Please, execute as 'python -m unittest tests/test_NN_gemmconv.py'")

# Default parameters
b = 1  # 32  # Batch size
c = 1  # 3  # Channels per layer
h = 6  # 128  # Layers height
w = 6  # 128  # Layers width
kn = 2  # 16  # Number of filters
kh = 4  # 16  # Filters height
kw = 4  # 16  # Filters width
padding = 2  # Padding
stride = 1  # Stride


class TestGemmConv(unittest.TestCase):

    def test_defaults(self):
        """
        Test that the default parameters lead to the same solution
        """
        layers = np.random.rand(b, c, h, w).astype(np.float32)
        filters = np.random.rand(kn, c, kh, kw).astype(np.float32, order='F')
        gemm_conv = GemmConv()
        gemm_conv_result = gemm_conv.gemm_conv(filters, layers,
                                               vpadding=padding, hpadding=padding,
                                               vstride=stride, hstride=stride)
        a_t = im2col_cython(layers, kh, kw, padding, padding, stride, stride)
        w_c = filters.reshape(kn, -1)
        im2col_mm_result = w_c @ a_t
        print("----------------")
        print("gemm_conv_result")
        print("----------------")
        print(gemm_conv_result)
        print("----------------")
        print("im2col_mm_result")
        print("----------------")
        print(im2col_mm_result)
        self.assertTrue(np.allclose(gemm_conv_result, im2col_mm_result))


if __name__ == '__main__':
    try:
        GemmConv()
    except NameError:
        sys.exit(-1)
    unittest.main()
