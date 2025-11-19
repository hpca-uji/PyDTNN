"""
Performance tests for transposing matrices

For running the tests run:
    python best_conv2d_transpose_0312.py
"""

from pydtnn.profilers.best_of_profiler import BestOfProfiler
from pydtnn.tests.abstract.common import alexnet_layers
from pydtnn.utils.best_transpose_0312 import best_transpose_0312
from pydtnn.utils import random


def main():
    layers = alexnet_layers
    bop = BestOfProfiler("Transpose 0312 comparison", best_transpose_0312)
    for layer in layers:
        d0, d1, d2, d3 = layer.shape
        original = random.random((d0, d1, d2, d3)).astype(layer.dtype, order="C", copy=None)
        bop(original)
    bop.print_results()


if __name__ == "__main__":
    main()
