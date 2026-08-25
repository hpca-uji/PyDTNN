"""Common methods and properties for various unitary tests"""

import gc
import logging
import sys
import unittest
import warnings

import numpy as np

from pydtnn.model.base import Base
from pydtnn.utils import rand
from pydtnn.utils.tensor import TensorFormat

__all__ = (
    "D",
    "Params",
    "TestCase",
    "verbose_test",
)

logger = logging.getLogger(__name__)


# @warning: must be a function, don't use a @property decorator
def verbose_test() -> bool:
    """Returns True if unittest has been called with -v or --verbose options."""
    return "-v" in sys.argv or "--verbose" in sys.argv


class Params(Base):
    """Configuration parameters for test execution."""

    def __init__(self) -> None:
        """Initializes default test parameters."""
        self.parallel_data = False
        self.dtype: np.dtype = np.dtype(np.float32)
        self.tensor_format = TensorFormat.NHWC
        self.backend = "cpu"
        self.batch_size = 8
        self.model_name: str = ""
        self.dataset_name = "synthetic"
        self.synthetic_train_samples = 128
        self.synthetic_test_samples = 128
        self.synthetic_input_shape = (3, 32, 32)
        self.synthetic_output_shape = (10,)


class TestCase(unittest.TestCase):
    """Base test case class for PyDTNN unit tests."""

    def setUp(self) -> None:
        """Initializes the test environment with fixed seeds and warning filters."""
        super().setUp()
        rand.seed(0)
        warnings.simplefilter("error")

    def tearDown(self) -> None:
        """Resets warning filters after test completion."""
        warnings.resetwarnings()
        gc.collect()
        super().tearDown()


class D:
    """Container for convolution layer dimensions and parameters."""

    def __init__(
        self,
        b: int = 1,
        c: int = 1,
        h: int = 128,
        w: int = 100,
        kn: int = 1,
        kh: int = 16,
        kw: int = 10,
        vpadding: int = 1,
        hpadding: int = 1,
        vstride: int = 1,
        hstride: int = 1,
        vdilation: int = 1,
        hdilation: int = 1,
        dtype: np.dtype = np.dtype(np.float32),
    ) -> None:
        """Initializes convolution dimensions and hyperparameters."""
        self.b = b  # Batch size
        self.c = c  # Channels per layer
        self.h = h  # Layers height
        self.w = w  # Layers width
        self.kn = kn  # Number of filters
        self.kh = kh  # Filters weights height
        self.kw = kw  # Filters weights width
        self.vpadding = vpadding  # Vertical padding
        self.hpadding = hpadding  # Horizontal padding
        self.vstride = vstride  # Vertical stride
        self.hstride = hstride  # Horizontal stride
        self.vdilation = vdilation  # Vertical dilation
        self.hdilation = hdilation  # Horizontal dilation
        self.dtype = dtype

    @property
    def ho(self) -> int:
        """Calculates the output height."""
        return (self.h + 2 * self.vpadding - self.vdilation * (self.kh - 1) - 1) // self.vstride + 1

    @property
    def wo(self) -> int:
        """Calculates the output width."""
        return (self.w + 2 * self.hpadding - self.hdilation * (self.kw - 1) - 1) // self.hstride + 1

    @property
    def shape(self) -> tuple[int, int, int, int]:
        """Returns the input shape as a tuple (b, c, h, w)."""
        return self.b, self.c, self.h, self.w

    def __str__(self) -> str:
        """Returns a formatted string representation of the layer dimensions."""
        return f"""\
x, weights, and y parameters:
  (b, c, h, w)    = {self.b} {self.c} {self.h} {self.w}
  (kn, c, kh, kw) = {self.kn} {self.c} {self.kh} {self.kw}
  (kn, b, ho, wo) = {self.kn} {self.b} {self.ho} {self.wo}
  padding         = {self.vpadding} {self.hpadding}
  stride          = {self.vstride} {self.hstride}
  dilation        = {self.vdilation} {self.hdilation}
"""


alexnet_layers = [
    # AlexNet Cifar
    D(64, 3, 32, 32, 64, 3, 3, 1, 1, 2, 2, 1, 1),
    D(64, 64, 8, 8, 192, 3, 3, 1, 1, 1, 1, 1, 1),
    D(64, 192, 4, 4, 384, 3, 3, 1, 1, 1, 1, 1, 1),
    D(64, 384, 4, 4, 256, 3, 3, 1, 1, 1, 1, 1, 1),
    D(64, 256, 4, 4, 256, 3, 3, 1, 1, 1, 1, 1, 1),
    # AlexNet ImageNet
    D(64, 3, 227, 227, 96, 11, 11, 1, 1, 4, 4, 1, 1),
    D(64, 96, 27, 27, 256, 5, 5, 1, 1, 1, 1, 1, 1),
    D(64, 256, 13, 13, 384, 3, 3, 1, 1, 1, 1, 1, 1),
    D(64, 384, 13, 13, 384, 3, 3, 1, 1, 1, 1, 1, 1),
    D(64, 384, 13, 13, 256, 3, 3, 1, 1, 1, 1, 1, 1),
]

alexnet_backward_layers = []
for layer in alexnet_layers:
    # w <- y (kn * b * ho * wo)
    alexnet_backward_layers.append(
        D(
            layer.c,
            layer.b,
            layer.h,
            layer.w,
            layer.kn,
            layer.ho,
            layer.wo,
            layer.vpadding,
            layer.hpadding,
            layer.vstride,
            layer.hstride,
        )
    )

alexnet_all_layers = alexnet_layers + alexnet_backward_layers
