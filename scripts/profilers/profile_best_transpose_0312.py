"""
Performance tests for transposing matrices

For running the tests run:
    python best_conv2d_transpose_0312.py
"""

from pydtnn.tests.abstract.common import alexnet_layers
from pydtnn.utils import rand
from pydtnn.utils.best_of.best_transpose_0312 import best_transpose_0312
from pydtnn.utils.best_of_profiler import BestOfProfiler


def main() -> None:
    """Executes performance profiling for the best_transpose_0312 implementation across all defined AlexNet layers."""
    layers = alexnet_layers
    bop = BestOfProfiler("Transpose 0312 comparison", best_transpose_0312)  # type: ignore
    for layer in layers:
        d0, d1, d2, d3 = layer.shape
        original = rand.random((d0, d1, d2, d3)).astype(layer.dtype)
        bop(original)
    bop.print_results()


if __name__ == "__main__":
    main()
