import inspect

import numpy as np

from pydtnn.backends.cpu.utils.im2row_nhwc_cython import im2row_nhwc_cython
from pydtnn.tests.abstract.common import verbose_test, D, alexnet_layers, TestCase
from pydtnn.utils import print_with_header, random


class ConvCommonTestCase(TestCase):
    """
    Tests that conv leads to the same results as i2c and mm.
    """

    @classmethod
    def _compute_both(cls, weights: np.ndarray, x: np.ndarray, biases: np.ndarray | None = None,
                      vpadding=0, hpadding=0, vstride=1, hstride=1,
                      vdilation=1, hdilation=1) -> tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError()

    @staticmethod
    def _get_config() -> D:
        return D()

    @staticmethod
    def _compute(weights: np.ndarray, x: np.ndarray, biases: np.ndarray | None = None, kh=1, kw=1, vpadding=0, hpadding=0, vstride=1, hstride=1, vdilation=1, hdilation=1):
        raise NotImplementedError()

    def test_raise_on_different_strides(self):
        d = self._get_config()
        weights = np.ones((d.c, d.kh, d.kw, d.kn)).astype(np.float32, order='C')
        x = np.ones((d.b, d.h, d.w, d.c)).astype(np.float32, order='C')
        test_result, ref_result = self._compute_both(weights, x,
                                                     vpadding=d.vpadding, hpadding=d.hpadding,
                                                     vstride=1, hstride=2,
                                                     vdilation=d.vdilation, hdilation=d.hdilation)
        self.assertTrue(np.allclose(test_result, ref_result))

    def test_defaults_with_ones(self):
        """
        Test that the default parameters on ones matrices lead to the same solution
        """
        d = self._get_config()
        weights = np.ones((d.c, d.kh, d.kw, d.kn)).astype(np.float32, order='C')
        x = np.ones((d.b, d.h, d.w, d.c)).astype(np.float32, order='C')
        test_result, ref_result = self._compute_both(weights, x,
                                                     vpadding=d.vpadding, hpadding=d.hpadding,
                                                     vstride=d.vstride, hstride=d.hstride,
                                                     vdilation=d.vdilation, hdilation=d.hdilation)
        self.assertTrue(np.allclose(test_result, ref_result))

    def test_defaults_with_random(self):
        """
        Test that the default parameters on random matrices lead to the same solution
        """
        d = self._get_config()
        weights = random.random((d.c, d.kh, d.kw, d.kn)).astype(np.float32, order='C')
        x = random.random((d.b, d.h, d.w, d.c)).astype(np.float32, order='C')
        test_result, ref_result = self._compute_both(weights, x,
                                                     vpadding=d.vpadding, hpadding=d.hpadding,
                                                     vstride=d.vstride, hstride=d.hstride,
                                                     vdilation=d.vdilation, hdilation=d.hdilation)
        self.assertTrue(np.allclose(test_result, ref_result))

    def test_defaults_including_biases_with_random(self):
        """
        Test that the default parameters on random matrices, including b, lead to the same solution
        """
        d = self._get_config()
        weights = random.random((d.c, d.kh, d.kw, d.kn)).astype(np.float32, order='C')
        x = random.random((d.b, d.h, d.w, d.c)).astype(np.float32, order='C')
        ho = (d.h + 2 * d.vpadding - d.vdilation * (d.kh - 1) - 1) // d.vstride + 1
        wo = (d.w + 2 * d.hpadding - d.hdilation * (d.kw - 1) - 1) // d.hstride + 1
        #biases = random.random((d.b, ho, wo, d.kn)).astype(np.float32, order='C')
        biases = random.random((d.kn, )).astype(np.float32, order='C')
        test_result, ref_result = self._compute_both(weights, x, biases=biases,
                                                     vpadding=d.vpadding, hpadding=d.hpadding,
                                                     vstride=d.vstride, hstride=d.hstride,
                                                     vdilation=d.vdilation, hdilation=d.hdilation)
        diff = test_result - ref_result
        self.assertTrue(np.allclose(test_result, ref_result), f"The difference is to big (rtol=1.e-5, atol=1.e-8). max diff: {diff.max()}. min diff: {diff.min()}")

    def test_with_different_kn(self):
        d = self._get_config()
        if verbose_test():
            print_with_header("{}".format(inspect.stack()[1][3]), None)
            print(" kn   Maximum difference    sum(cg_result)")
            print("----+--------------------+-----------------")
        x = random.random((d.b, d.h, d.w, d.c)).astype(np.float32, order='C')
        np_all_close_for_all_cases = True
        for kn in range(1, 32):
            weights = random.random((d.c, d.kh, d.kw, kn)).astype(np.float32, order='C')
            test_result: np.ndarray = self._compute(weights, x,
                                                    kw=d.kw, kh=d.kh,
                                                    vpadding=d.vpadding, hpadding=d.hpadding,
                                                    vstride=d.vstride, hstride=d.hstride,
                                                    vdilation=d.vdilation, hdilation=d.hdilation)
            test_result: np.ndarray = test_result.reshape((-1, kn), copy=False)
            n, h, w, c = x.shape

            ho = (h + 2 * d.vpadding - d.vdilation * (d.kh - 1) - 1) // d.vstride + 1
            wo = (w + 2 * d.hpadding - d.hdilation * (d.kw - 1) - 1) // d.hstride + 1

            dim_n = n * ho * wo
            dim_c = c * d.kh * d.kw

            x_c = np.zeros(shape=(dim_n, dim_c), dtype=x.dtype)

            # FIXME: To abstract method (or use compute both)
            im2row_nhwc_cython(x, x_c,
                               d.kh, d.kw, ho, wo,
                               d.vpadding, d.hpadding,
                               d.vstride, d.hstride,
                               d.vdilation, d.hdilation)
            w_c = weights.reshape((-1, kn), copy=False)
            ref_result = x_c @ w_c
            if verbose_test():
                print("{:3}    {:9.7f}             {:11.2f}"
                      "".format(kn, max([abs(x - y) for x, y in zip(test_result.flatten(),
                                                                    ref_result.flatten())]),
                                np.sum(test_result)))
            np_all_close_for_all_cases = np_all_close_for_all_cases and np.allclose(test_result,
                                                                                    ref_result)
        self.assertTrue(np_all_close_for_all_cases)

    def test_with_different_b(self):
        d = self._get_config()
        if verbose_test():
            print_with_header("{}".format(inspect.stack()[1][3]), None)
            print("  b   Maximum difference")
            print("----+--------------------")
        weights = random.random((d.c, d.kh, d.kw, d.kn)).astype(np.float32, order='C')
        np_all_close_for_all_cases = True
        for b in range(1, 32):
            x = random.random((b, d.h, d.w, d.c)).astype(np.float32, order='C')
            test_result: np.ndarray = self._compute(weights, x,
                                                    kw=d.kw, kh=d.kh,
                                                    vpadding=d.vpadding, hpadding=d.hpadding,
                                                    vstride=d.vstride, hstride=d.hstride,
                                                    vdilation=d.vdilation, hdilation=d.hdilation)
            test_result: np.ndarray = test_result.reshape((-1, d.kn), copy=False)
            n, h, w, c = x.shape

            ho = (h + 2 * d.vpadding - d.vdilation * (d.kh - 1) - 1) // d.vstride + 1
            wo = (w + 2 * d.hpadding - d.hdilation * (d.kw - 1) - 1) // d.hstride + 1

            dim_n = n * ho * wo
            dim_c = c * d.kh * d.kw

            x_c = np.zeros(shape=(dim_n, dim_c), dtype=x.dtype)

            # FIXME: To abstract method (or use compute both)
            im2row_nhwc_cython(x, x_c,
                               d.kh, d.kw, ho, wo,
                               d.vpadding, d.hpadding,
                               d.vstride, d.hstride,
                               d.vdilation, d.hdilation)
            w_c = weights.reshape((-1, d.kn), copy=False)
            ref_result = x_c @ w_c
            if verbose_test():
                print("{:3}    {:9.7f}".format(b,
                                               max([abs(x - y) for x, y
                                                    in
                                                    zip(test_result.flatten(), ref_result.flatten())])))
            np_all_close_for_all_cases = np_all_close_for_all_cases and np.allclose(test_result,
                                                                                    ref_result)
        self.assertTrue(np_all_close_for_all_cases)

    def test_with_different_padding(self):
        d = self._get_config()
        if verbose_test():
            print_with_header("{}".format(inspect.stack()[1][3]), None)
            print("  p   Maximum difference")
            print("----+--------------------")
        weights = random.random((d.c, d.kh, d.kw, d.kn)).astype(np.float32, order='C')
        x = random.random((d.b, d.h, d.w, d.c)).astype(np.float32, order='C')
        np_all_close_for_all_cases = True
        for padding in range(0, 5):
            test_result: np.ndarray = self._compute(weights, x,
                                                    kw=d.kw, kh=d.kh,
                                                    vpadding=padding, hpadding=padding,
                                                    vstride=d.vstride, hstride=d.hstride,
                                                    vdilation=d.vdilation, hdilation=d.hdilation)
            test_result: np.ndarray = test_result.reshape((-1, d.kn), copy=False)
            n, h, w, c = x.shape

            ho = (h + 2 * padding - d.vdilation * (d.kh - 1) - 1) // d.vstride + 1
            wo = (w + 2 * padding - d.hdilation * (d.kw - 1) - 1) // d.hstride + 1

            dim_n = n * ho * wo
            dim_c = c * d.kh * d.kw

            x_c = np.zeros(shape=(dim_n, dim_c), dtype=x.dtype)

            # FIXME: To abstract method (or use compute both)
            im2row_nhwc_cython(x, x_c,
                               d.kh, d.kw, ho, wo,
                               padding, padding,
                               d.vstride, d.hstride,
                               d.vdilation, d.hdilation)
            w_c = weights.reshape((-1, d.kn), copy=False)
            ref_result = x_c @ w_c
            if verbose_test():
                print("{:3}    {:9.7f}".format(padding,
                                               max([abs(x - y) for x, y
                                                    in
                                                    zip(test_result.flatten(), ref_result.flatten())])))
            np_all_close_for_all_cases = np_all_close_for_all_cases and np.allclose(test_result,
                                                                                    ref_result)
        self.assertTrue(np_all_close_for_all_cases)

    def test_with_different_stride(self):
        d = self._get_config()
        if verbose_test():
            print_with_header("{}".format(inspect.stack()[1][3]), None)
            print("  s   Maximum difference")
            print("----+--------------------")
        weights = random.random((d.c, d.kh, d.kw, d.kn)).astype(np.float32, order='C')
        x = random.random((d.b, d.h, d.w, d.c)).astype(np.float32, order='C')
        np_all_close_for_all_cases = True
        for stride in range(1, 6):
            test_result: np.ndarray = self._compute(weights, x,
                                                    kw=d.kw, kh=d.kh,
                                                    vpadding=d.vpadding, hpadding=d.hpadding,
                                                    vstride=stride, hstride=stride,
                                                    vdilation=d.vdilation, hdilation=d.hdilation)
            test_result: np.ndarray = test_result.reshape((-1, d.kn), copy=False)
            n, h, w, c = x.shape

            ho = (h + 2 * d.vpadding - d.vdilation * (d.kh - 1) - 1) // stride + 1
            wo = (w + 2 * d.hpadding - d.hdilation * (d.kw - 1) - 1) // stride + 1

            dim_n = n * ho * wo
            dim_c = c * d.kh * d.kw

            x_c = np.zeros(shape=(dim_n, dim_c), dtype=x.dtype)

            # FIXME: To abstract method (or use compute both)
            im2row_nhwc_cython(x, x_c,
                               d.kh, d.kw, ho, wo,
                               d.vpadding, d.hpadding,
                               stride, stride,
                               d.vdilation, d.hdilation)
            w_c = weights.reshape((-1, d.kn), copy=False)
            ref_result = x_c @ w_c
            if verbose_test():
                print("{:3}    {:9.7f}".format(stride,
                                               max([abs(x - y) for x, y
                                                    in
                                                    zip(test_result.flatten(), ref_result.flatten())])))
            np_all_close_for_all_cases = np_all_close_for_all_cases and np.allclose(test_result,
                                                                                    ref_result)
        self.assertTrue(np_all_close_for_all_cases)

    def test_with_different_strides(self):
        d = self._get_config()
        if verbose_test():
            print_with_header("{}".format(inspect.stack()[1][3]), None)
            print(" vs  hs   Maximum difference")
            print("--------+--------------------")
        weights = random.random((d.c, d.kh, d.kw, d.kn)).astype(np.float32, order='C')
        x = random.random((d.b, d.h, d.w, d.c)).astype(np.float32, order='C')
        for vstride in range(1, 5):
            for hstride in range(1, 5):
                if vstride == hstride:
                    continue
                test_result: np.ndarray = self._compute(weights, x,
                                                        kw=d.kw, kh=d.kh,
                                                        vpadding=d.vpadding, hpadding=d.hpadding,
                                                        vstride=vstride, hstride=hstride,
                                                        vdilation=d.vdilation, hdilation=d.hdilation)
                test_result: np.ndarray = test_result.reshape((-1, d.kn), copy=False)
                n, h, w, c = x.shape

                ho = (h + 2 * d.vpadding - d.vdilation * (d.kh - 1) - 1) // vstride + 1
                wo = (w + 2 * d.hpadding - d.hdilation * (d.kw - 1) - 1) // hstride + 1

                dim_n = n * ho * wo
                dim_c = c * d.kh * d.kw

                x_c = np.zeros(shape=(dim_n, dim_c), dtype=x.dtype)

                # FIXME: To abstract method (or use compute both)
                im2row_nhwc_cython(x, x_c,
                                   d.kh, d.kw, ho, wo,
                                   d.vpadding, d.hpadding,
                                   vstride, hstride,
                                   d.vdilation, d.hdilation)
                w_c = weights.reshape((-1, d.kn), copy=False)
                ref_result = x_c @ w_c
                if verbose_test():
                    print("{:3} {:3}    {:9.7f}".format(vstride, hstride,
                                                        max([abs(x - y) for x, y
                                                             in
                                                             zip(test_result.flatten(),
                                                                 ref_result.flatten())])))
                self.assertTrue(np.allclose(test_result, ref_result),
                                f"Results differ with vstride {vstride} and hstride {hstride}")

    def test_with_different_dilation(self):
        d = self._get_config()
        if verbose_test():
            print_with_header("{}".format(inspect.stack()[1][3]), None)
            print("  s   Maximum difference")
            print("----+--------------------")
        weights = random.random((d.c, d.kh, d.kw, d.kn)).astype(np.float32, order='C')
        x = random.random((d.b, d.h, d.w, d.c)).astype(np.float32, order='C')
        np_all_close_for_all_cases = True
        for dilation in range(1, 3):
            test_result: np.ndarray = self._compute(weights, x,
                                                    kw=d.kw, kh=d.kh,
                                                    vpadding=d.vpadding, hpadding=d.hpadding,
                                                    vstride=d.vstride, hstride=d.hstride,
                                                    vdilation=dilation, hdilation=dilation)
            test_result: np.ndarray = test_result.reshape((-1, d.kn), copy=False)
            n, h, w, c = x.shape

            ho = (h + 2 * d.vpadding - dilation * (d.kh - 1) - 1) // d.vstride + 1
            wo = (w + 2 * d.hpadding - dilation * (d.kw - 1) - 1) // d.hstride + 1

            dim_n = n * ho * wo
            dim_c = c * d.kh * d.kw

            x_c = np.zeros(shape=(dim_n, dim_c), dtype=x.dtype)

            # FIXME: To abstract method (or use compute both)
            im2row_nhwc_cython(x, x_c,
                               d.kh, d.kw, ho, wo,
                               d.vpadding, d.hpadding,
                               d.vstride, d.hstride,
                               dilation, dilation)
            w_c = weights.reshape((-1, d.kn), copy=False)
            ref_result = x_c @ w_c
            if verbose_test():
                print("{:3}    {:9.7f}".format(dilation,
                                               max([abs(x - y) for x, y
                                                    in
                                                    zip(test_result.flatten(), ref_result.flatten())])))
            np_all_close_for_all_cases = np_all_close_for_all_cases and np.allclose(test_result,
                                                                                    ref_result)
        self.assertTrue(np_all_close_for_all_cases)

    def test_alexnet_layers(self):
        if verbose_test():
            print_with_header("{}".format(inspect.stack()[1][3]), None)
            print(" layer   Maximum difference")
            print("-------+--------------------")
        layers = alexnet_layers
        for n, layer in enumerate(layers):
            weights = random.random((layer.c, layer.kh, layer.kw, layer.kn)).astype(np.float32, order='C')
            x = random.random((layer.b, layer.h, layer.w, layer.c)).astype(np.float32, order='C')
            test_result: np.ndarray = self._compute(weights, x,
                                                    kw=layer.kw, kh=layer.kh,
                                                    vpadding=layer.vpadding, hpadding=layer.hpadding,
                                                    vstride=layer.vstride, hstride=layer.hstride,
                                                    vdilation=layer.vdilation, hdilation=layer.hdilation)
            test_result: np.ndarray = test_result.reshape(-1, layer.kn)
            n, h, w, c = x.shape

            dim_n = n * layer.ho * layer.wo
            dim_c = c * layer.kh * layer.kw

            x_c = np.zeros(shape=(dim_n, dim_c), dtype=x.dtype)

            # FIXME: To abstract method (or use compute both)
            im2row_nhwc_cython(x, x_c,
                               layer.kh, layer.kw, layer.ho, layer.wo,
                               layer.vpadding, layer.hpadding,
                               layer.vstride, layer.hstride,
                               layer.vdilation, layer.hdilation)
            w_c = weights.reshape(-1, layer.kn)
            ref_result = x_c @ w_c
            if verbose_test():
                print("   {:2}      {:9.7f}".format(n,
                                                    max([abs(x - y) for x, y
                                                         in
                                                         zip(test_result.flatten(),
                                                             ref_result.flatten())])))
                if n == 9:
                    print("Flags for last test_result output:")
                    print(test_result.flags)
            self.assertTrue(np.allclose(test_result, ref_result),
                            f"Results differ for AlexNet Cifar and ImageNet layers number {n}")
