"""
Performance tests for transposing matrices

For running the tests run:
    python best_conv2d_transpose_0231.py
"""

import numpy as np

from pydtnn.profilers.best_of_profiler import BestOfProfiler
from pydtnn.tests.common import alexnet_layers
from pydtnn.utils.best_transpose_1023 import best_transpose_1023


def main():
    layers = alexnet_layers
    bop = BestOfProfiler("Transpose 1023 comparison", best_transpose_1023)
    for layer in layers:
        d0, d1, d2, d3 = layer.shape
        original = np.random.rand(d0, d1, d2, d3).astype(layer.dtype, order="C")
        bop(original)
    bop.print_results()


if __name__ == '__main__':
    main()
