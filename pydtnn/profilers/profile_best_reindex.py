"""
Performance tests for reindex matrices

For running the tests run:
    python best_conv2d_reindex.py

"""

import numpy as np

from pydtnn.backends.cpu.layers.conv_2d_cpu import Conv2DCPU
from pydtnn.profilers.best_of_profiler import BestOfProfiler
from pydtnn.tests.common import alexnet_layers
from pydtnn.utils.best_reindex import best_reindex


class Conv2DCPUWrapper(Conv2DCPU):
    """
    Used to access protected member _get_x_new_indexes_and_xstride()
    """

    def _backward_depthwise(self, dy):
        pass

    def _backward_pointwise(self, dy):
        pass

    @staticmethod
    def get_x_new_indexes_and_xstride(kx, xo, s):
        return Conv2DCPUWrapper._get_x_new_indexes_and_xstride(kx, xo, s)


def main():
    layers = alexnet_layers
    # Keep only those layers with vstride != 1 or hstride != 1
    layers = [la for la in layers if la.vstride != 1 or la.hstride != 1]
    # Increase x (h, w) dimensions by 2*vpadding, 2* hpadding
    for la in layers:
        la.h = la.h + 2 * la.vpadding
        la.vpadding = 0
        la.w = la.w + 2 * la.hpadding
        la.hpadding = 0
    bop = BestOfProfiler("Reindex comparison", best_reindex)
    for layer in layers:
        w_shape = (layer.kn, layer.c, layer.kh, layer.kw)
        x_shape = (layer.b, layer.c, layer.h, layer.w)
        y_shape = (layer.kn, layer.b, layer.ho, layer.wo)
        kn, c, kh, kw = w_shape
        # b, c, h, w = x_shape
        kn, b, ho, wo = y_shape
        h_new_indexes, cg_vstride = Conv2DCPUWrapper.get_x_new_indexes_and_xstride(kh, ho, layer.vstride)
        v_new_indexes, cg_hstride = Conv2DCPUWrapper.get_x_new_indexes_and_xstride(kw, wo, layer.hstride)
        matrix_in = np.random.rand(*x_shape).astype(dtype=np.float32)
        bop(matrix_in, h_new_indexes, v_new_indexes)
    bop.print_results()


if __name__ == '__main__':
    main()
