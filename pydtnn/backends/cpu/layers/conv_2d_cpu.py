#
#  This file is part of Python Distributed Training of Neural Networks (PyDTNN)
#
#  Copyright (C) 2021-22 Universitat Jaume I
#
#  PyDTNN is free software: you can redistribute it and/or modify it under the
#  terms of the GNU General Public License as published by the Free Software
#  Foundation, either version 3 of the License, or (at your option) any later
#  version.
#
#  This program is distributed in the hope that it will be useful, but WITHOUT
#  ANY WARRANTY; without even the implied warranty of MERCHANTABILITY
#  or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public
#  License for more details.
#
#  You should have received a copy of the GNU General Public License along
#  with this program. If not, see <https://www.gnu.org/licenses/>.
#

import sys

from pydtnn.backends.cpu.layers import LayerCPU
from pydtnn.backends.cpu.layers.conv_2d_variants.best_of_variant import BestOfVariant
from pydtnn.backends.cpu.layers.conv_2d_variants.conv_gemm_variant import ConvGemmVariant
from pydtnn.backends.cpu.layers.conv_2d_variants.depthwise_variant import DepthwiseVariant
from pydtnn.backends.cpu.layers.conv_2d_variants.pointwise_variant import PointwiseVariant
from pydtnn.performance_models import im2col_time, matmul_time, col2im_time
from pydtnn.utils import PYDTNN_TENSOR_FORMAT_NCHW


class Conv2DCPU(LayerCPU,
                DepthwiseVariant,
                PointwiseVariant,
                # I2CVariant (provided from ConvWinogradVariant)
                ConvGemmVariant,
                # ConvWinogradVariant (provided from BestOfVariant)
                # ConvDirectVariant (provided from BestOfVariant)
                BestOfVariant):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Other parameters initialized in initialize()
        self.variant = None
        self.weights = None
        self.biases = None
        self.fwd_time = None
        self.bwd_time = None

    def initialize(self, prev_shape, need_dx=True):
        super().initialize(prev_shape, need_dx)
        # Weights
        self.weights = self.weights_initializer(self.weights_shape, self.model.dtype)
        # Biases
        if self.use_bias:
            self.biases = self.biases_initializer((self.co,), self.model.dtype)
        # Select variants if it has not been already selected (e.g., by BestOfVariant)
        if self.variant is None:
            # Select variant when best_of is not enabled
            variant = 'i2c'  # Default variant is i2c
            if self.grouping == 'pointwise':  # 2nd alternative
                variant = 'pointwise'
            elif self.grouping == 'depthwise':  # 3rd alternative
                variant = 'depthwise'
            else:  # 4th alternative: one of convWinograd, convGemm or convDirect
                # Check colliding options
                if self.model.enable_conv_winograd and self.model.enable_conv_direct:
                    sys.stderr.write("Error: please, select exactly one of conv_winograd or conv_direct")
                    sys.exit(1)
                if self.model.enable_conv_gemm and self.model.enable_conv_direct:
                    sys.stderr.write("Error: please, select exactly one of conv_gemm or conv_direct")
                    sys.exit(1)
                if self.model.enable_conv_winograd:
                    # If conv_winograd is enabled, use it if it is possible.
                    # If not, fallback to cg or i2c variant depending on the user's choice.
                    if self.cw_constraints_fulfilled:
                        variant = 'cw'
                    elif self.model.enable_conv_gemm:
                        variant = 'cg'
                    elif self.model.enable_conv_direct:
                        variant = 'cd0'
                    else:
                        variant = 'i2c'  # Redundant, just to make it clear
                elif self.model.enable_conv_gemm:  # After conv_winograd, as it can be selected as fallback
                    variant = 'cg'
                elif self.model.enable_conv_direct:
                    variant = 'cd0'
            self.variant = variant
        # Set forward and backward implementations based on self.variant
        forward, backward = self._get_forward_and_backward(self.variant)
        setattr(self, "forward", forward)
        setattr(self, "backward", backward)
        # Performance models
        self.fwd_time = \
            im2col_time(m=(self.ci * self.kh * self.kw), n=(self.model.batch_size * self.ho * self.wo),
                        cpu_speed=self.model.cpu_speed, memory_bw=self.model.memory_bw,
                        dtype=self.model.dtype) + \
            matmul_time(m=self.co, n=(self.model.batch_size * self.ho * self.wo), k=(self.ci * self.kh * self.kw),
                        cpu_speed=self.model.cpu_speed, memory_bw=self.model.memory_bw,
                        dtype=self.model.dtype)
        self.bwd_time = \
            matmul_time(m=self.co, n=(self.ci * self.kh * self.kw), k=(self.model.batch_size * self.ho * self.wo),
                        cpu_speed=self.model.cpu_speed, memory_bw=self.model.memory_bw,
                        dtype=self.model.dtype)
        if need_dx:
            self.bwd_time += matmul_time(m=(self.ci * self.kh * self.kw), n=(self.model.batch_size * self.ho * self.wo),
                                         k=self.co, cpu_speed=self.model.cpu_speed,
                                         memory_bw=self.model.memory_bw, dtype=self.model.dtype)
        else:
            self.bwd_time += col2im_time(m=(self.ci * self.kh * self.kw), n=(self.model.batch_size * self.ho * self.wo),
                                         cpu_speed=self.model.cpu_speed, memory_bw=self.model.memory_bw,
                                         dtype=self.model.dtype)

    def forward(self, x):
        """This is a fake forward function. It will be masked on initialization by a _forward implementation"""
        pass

    def backward(self, dy):
        """This is a fake backward function. It will be masked on initialization by a _backward implementation"""
        pass

    def print_in_convdirect_format(self):
        if self.hstride != 1 or self.vstride != 1:
            return
        # #l kn wo ho t kh kw ci wi hi"
        if self.model.tensor_format == PYDTNN_TENSOR_FORMAT_NCHW:
            ci, hi, wi = self.prev_shape
        else:
            hi, wi, ci = self.prev_shape
        print(self.id, self.co, self.wo, self.ho, self.model.batch_size, self.kh, self.kw, ci, wi, hi, sep="\t")

    def _get_forward_and_backward(self, variant):
        tensor_format = 'nchw' if self.model.tensor_format == PYDTNN_TENSOR_FORMAT_NCHW else 'nhwc'
        return (getattr(self, f'_forward_{variant}_{tensor_format}'),
                getattr(self, f'_backward_{variant}_{tensor_format}'))
