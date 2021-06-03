"""
Performance tests for transposing matrices

For running the tests run:
    python best_conv2d_transpose_2d_f2c.py
"""

import numpy as np

from pydtnn.profilers.best_of_profiler import BestOfProfiler
from pydtnn.tests.common import alexnet_layers
from pydtnn.utils.best_transpose_2d_f2c import best_transpose_2d_fc


def main():
    layers = alexnet_layers
    bop = BestOfProfiler("Transpose 2D Fortran2C comparison", best_transpose_2d_fc)
    for layer in layers:
        d0, d1 = (layer.kn, layer.b * layer.ho * layer.wo)
        original = np.random.rand(d0, d1).astype(np.float32, order="F")
        bop(original)
    bop.print_results()


if __name__ == '__main__':
    main()
