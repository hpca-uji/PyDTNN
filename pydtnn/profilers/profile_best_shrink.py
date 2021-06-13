"""
Performance tests for shrinking matrices

For running the tests run:
    python best_conv2d_shrink.py

"""

import numpy as np

from pydtnn.profilers.best_of_profiler import BestOfProfiler
from pydtnn.tests.common import alexnet_all_layers
from pydtnn.utils.best_shrink import best_shrink


def main():
    layers = alexnet_all_layers
    bop = BestOfProfiler("Shrinking comparison", best_shrink)
    for layer in layers:
        x_shape = (layer.b, layer.c, layer.h, layer.w)
        matrix_in = np.random.rand(*x_shape).astype(dtype=np.float32)
        bop(matrix_in, layer.vpadding, layer.hpadding)
    bop.print_results()


if __name__ == '__main__':
    main()
