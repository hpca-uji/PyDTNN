"""
Performance profiling script for evaluating the best transpose 1023 implementation.

This script benchmarks the performance of the `best_transpose_1023` function
against standard AlexNet layer configurations using the `BestOfProfiler`.
"""

from pydtnn.tests.abstract.common import alexnet_layers
from pydtnn.utils import random
from pydtnn.utils.best_of.best_transpose_1023 import best_transpose_1023
from pydtnn.utils.best_of_profiler import BestOfProfiler


def main():
    """
    Executes the profiling routine for the transpose 1023 operation.

    Iterates through predefined AlexNet layer shapes, generates random input
    tensors, profiles the execution time using BestOfProfiler, and outputs
    the final performance results.
    """
    layers = alexnet_layers
    bop = BestOfProfiler("Transpose 1023 comparison", best_transpose_1023)
    for layer in layers:
        d0, d1, d2, d3 = layer.shape
        original = random.random((d0, d1, d2, d3)).astype(layer.dtype)
        bop(original)
    bop.print_results()


if __name__ == "__main__":
    main()
