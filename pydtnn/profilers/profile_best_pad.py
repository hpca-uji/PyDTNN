"""
Performance tests for padding matrices

For running the tests run:
    python best_conv2d_pad.py

"""

import numpy as np

from pydtnn.profilers.best_of_profiler import BestOfProfiler
from pydtnn.tests.common import alexnet_all_layers
from pydtnn.utils.best_pad import best_pad


def main():
    layers = alexnet_all_layers
    bop = BestOfProfiler("Padding comparison", best_pad)
    for layer in layers:
        x_shape = (layer.b, layer.c, layer.h, layer.w)
        matrix_in = np.random.rand(*x_shape).astype(dtype=np.float32)
        bop(matrix_in, layer.vpadding, layer.hpadding)
    bop.print_results()


if __name__ == '__main__':
    main()
