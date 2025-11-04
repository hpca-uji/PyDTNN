import inspect
import sys
import unittest

import numpy as np

from pydtnn.activations.relu import Relu
from pydtnn.backends.cpu.activations.relu_cpu import ReluCPU
from pydtnn.backends.cpu.layers.concatenation_block_cpu import ConcatenationBlockCPU
from pydtnn.backends.cpu.layers.conv_2d_relu_cpu import Conv2DReluCPU
from pydtnn.layers.batch_normalization import BatchNormalization
from pydtnn.layers.concatenation_block import ConcatenationBlock
from pydtnn.layers.conv_2d import Conv2D
from pydtnn.layers.conv_2d_batch_normalization_relu import Conv2DBatchNormalizationRelu
from pydtnn.layers.conv_2d_relu import Conv2DRelu
from pydtnn.model import Model
from pydtnn.backends.cpu.layers.conv_2d_cpu import Conv2DCPU
from pydtnn.tests.common import verbose_test, D
from pydtnn.tests.common import Params, TestCase
from pydtnn.tests.conv2d_conv_gemm import Conv2DConvGemmTestCase as _Conv2DConvGemmTestCase
from pydtnn.utils.tensor import TensorFormat
from pydtnn.utils import print_with_header, random
from pydtnn.utils.initializers import glorot_uniform, zeros


class Conv2DBatchNormalizationReluTestCase(_Conv2DConvGemmTestCase):
    """
    Tests that Conv2D+BatchNormalization+Relu leads to the same results than Conv2DBatchNormalizationRelu
    """

    @staticmethod
    def _get_layers(d: D, deconv=False, trans=False) -> tuple[Conv2DCPU, Conv2DCPU]:
        params = Params()
        params.tensor_format = TensorFormat.NHWC.upper()
        params.batch_size = d.b
        params.enable_conv_gemm = True
        model = Model(**vars(params))
        model.mode = Model.Mode.TRAIN

        conv2d = Conv2D(nfilters=d.kn, filter_shape=(d.kh, d.kw),
                        padding=(d.vpadding, d.hpadding),
                        stride=(d.vstride, d.hstride),
                        dilation=(d.vdilation, d.hdilation),
                        use_bias=True, weights_initializer=glorot_uniform, biases_initializer=zeros)
        bn = BatchNormalization()
        relu = Relu()
        chain = ConcatenationBlock([
            conv2d,
            bn,
            relu
        ])
        chain.set_backend(model._backend)
        chain.set_model(model)
        chain.initialize(prev_shape=(d.c, d.h, d.w))

        fuse = Conv2DBatchNormalizationRelu(from_parent=conv2d, from_parent2=bn)
        fuse.set_backend(model._backend)
        fuse.set_model(model)
        fuse.initialize(from_parent_dict=conv2d.__dict__, prev_shape=(d.c, d.h, d.w))

        # Set the same initial weights and biases to both layers
        fuse.weights = conv2d.weights.copy()
        fuse.biases = conv2d.biases.copy()

        return chain, fuse


if __name__ == '__main__':
    try:
        Conv2DCPU()
    except NameError:
        sys.exit(-1)
    unittest.main()
