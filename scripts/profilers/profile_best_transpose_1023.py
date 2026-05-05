"""
Performance tests for transposing matrices

For running the tests run:
    python best_conv2d_transpose_0231.py
"""

from pydtnn.tests.abstract.common import alexnet_layers
from pydtnn.utils import random
from pydtnn.utils.best_of.best_transpose_1023 import best_transpose_1023
from pydtnn.utils.best_of_profiler import BestOfProfiler


def main():
    layers = alexnet_layers
    bop = BestOfProfiler("Transpose 1023 comparison", best_transpose_1023)
    for layer in layers:
        d0, d1, d2, d3 = layer.shape
        original = random.random((d0, d1, d2, d3)).astype(layer.dtype)
        bop(original)
    bop.print_results()


if __name__ == "__main__":
    main()
