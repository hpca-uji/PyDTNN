import time
import itertools

import numpy as np

from pydtnn.tests.conv_2d_conv_gemm import D, Conv2DConvGemmTestCase
from pydtnn.utils import random
from pydtnn.libs.libmpi import MPI


class Conv2DConvGemmSlowTestCase(Conv2DConvGemmTestCase):
    """
    Tests that Conv2D with conv_gemm leads to the same results than Conv2d with mm and i2c.T (exhaustive version)
    """
    # NOTE: Delete parent test to prevent re-export and re-testing
    global Conv2DConvGemmTestCase
    del Conv2DConvGemmTestCase

    random.seed(0)  # type: ignore
    dtype = np.float32

    R = list(itertools.product(
        range(1, 4),   # kn
        range(1, 4),   # b
        range(8, 11),  # c
        range(8, 11),  # h
        range(8, 11),  # w
        range(2, 12),  # kh
        range(2, 12),  # kw
        range(0, 4),   # vp
        range(0, 4),   # hp
        range(1, 4),   # vs
        range(1, 4)    # hs
    ))

    X = random.random((
        4,   # b
        11,  # c
        11,  # h
        11   # w
    )).astype(dtype, order="C")

    W = random.random((
        4,   # kn
        11,  # c
        11,  # kh
        11,  # kw
    )).astype(dtype, order="C")

    def test_forward_backward_multiple_params(self):
        """Tests that different input matrices, paddings and strides, lead to the same solution"""
        start = time.time()

        comm = MPI.COMM_WORLD
        batch = self.R[comm.rank::comm.size]

        for i, params in enumerate(batch):
            self._test_forward_backward_multiple_params(*params)

            if comm.rank:
                continue

            i *= comm.size
            perc = (i + 1) / len(self.R)
            elapsed = time.time() - start
            remain = (1 - perc) * (elapsed / perc)
            m, s = divmod(remain, 60)
            h, m = divmod(m, 60)
            d, h = divmod(h, 24)
            print(f"{perc:.5%} (eta: {d:2.0f}d {h:2.0f}h {m:2.0f}m {s:2.0f}s)", end="\r")

    def _test_forward_backward_multiple_params(self, kn: int, b: int, c: int, h: int, w: int, kh: int, kw: int, vpadding: int, hpadding: int, vstride: int, hstride: int):
        d = D(kn=kn, b=b, c=c, h=h, w=w, kh=kh, kw=kw, vpadding=vpadding, hpadding=hpadding, vstride=vstride, hstride=hstride)

        if d.kh >= d.h + 1 or d.kw >= d.w + 1:
            return

        if d.b != 1 or d.c != 1:
            x = self.X[:d.b, :d.c, :d.h, :d.w].copy(order="C")
        else:
            x = np.concatenate([
                np.arange(b := (i + 1) * 100, b + d.w, dtype=np.float32)
                for i in range(d.h)
            ]).reshape((d.b, d.c, d.h, d.w))

        weights = self.W[:d.kn, :d.c, :d.kh, :d.kw].copy(order="C")

        self._test_forward_backward(d, x, weights)
