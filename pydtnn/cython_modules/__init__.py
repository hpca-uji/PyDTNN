#
#  This file is part of Python Distributed Training of Neural Networks (PyDTNN)
#
#  Copyright (C) 2021-25 Universitat Jaume I
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
#  with this program.  If not, see <https://www.gnu.org/licenses/>.
#

__all__ = (
    "add_cython",
    "argmax_cython",
    "average_pool_2d_fwd_nchw_cython", "average_pool_2d_bwd_nchw_cython",
    "average_pool_2d_fwd_nhwc_cython", "average_pool_2d_bwd_nhwc_cython",
    "bn_inference_cython", "bn_inference_nchw_cython",
    "bn_relu_inference_cython",
    "bn_training_fwd_cython", "bn_training_bwd_cython",
    "depthwise_conv_nchw_cython", "depthwise_conv_backward_nchw_cython",
    "depthwise_conv_nhwc_cython", "depthwise_conv_backward_nhwc_cython",
    "eltw_sum_cython",
    "im2col_1ch_nchw_cython", "col2im_1ch_nchw_cython",
    "im2col_nchw_cython", "col2im_nchw_cython",
    "im2row_1ch_nhwc_cython", "row2im_1ch_nhwc_cython",
    "im2row_nhwc_cython", "row2im_nhwc_cython",
    "max_pool_2d_fwd_nchw_cython", "max_pool_2d_bwd_nchw_cython",
    "max_pool_2d_fwd_nhwc_cython", "max_pool_2d_bwd_nhwc_cython",
    "pointwise_conv_cython",
    "relu_cython", "capped_relu_cython", "leaky_relu_cython",
    "transpose_0231_ikj_cython", "transpose_0231_ijk_cython", "transpose_0312_ijk_cython",
    "transpose_0312_ikj_cython", "transpose_1023_jik_cython", "transpose_1023_ijk_cython",
    "adaptive_avg_pooling_fwd_nchw_cython", "adaptive_avg_pooling_bwd_nchw_cython",
    "adaptive_avg_pooling_fwd_nhwc_cython", "adaptive_avg_pooling_bwd_nhwc_cython",
    "memoryview_index",
    "round",
    "sigmoid_fwd_cython", "sigmoid_bwd_cython", 
    "log_fwd_cython", "log_bwd_cython"
)

from pydtnn.cython_compiled_files.add_cython import add_cython
from pydtnn.cython_compiled_files.argmax_cython import argmax_cython
from pydtnn.cython_compiled_files.average_pool_2d_nchw_cython import average_pool_2d_fwd_nchw_cython, average_pool_2d_bwd_nchw_cython
from pydtnn.cython_compiled_files.average_pool_2d_nhwc_cython import average_pool_2d_fwd_nhwc_cython, average_pool_2d_bwd_nhwc_cython
from pydtnn.cython_compiled_files.bn_inference_cython import bn_inference_cython, bn_inference_nchw_cython, bn_relu_inference_cython
from pydtnn.cython_compiled_files.bn_training_cython import bn_training_fwd_cython, bn_training_bwd_cython
from pydtnn.cython_compiled_files.depthwise_conv_nchw_cython import depthwise_conv_nchw_cython, depthwise_conv_backward_nchw_cython
from pydtnn.cython_compiled_files.depthwise_conv_nhwc_cython import depthwise_conv_nhwc_cython, depthwise_conv_backward_nhwc_cython
from pydtnn.cython_compiled_files.eltw_sum_cython import eltw_sum_cython
from pydtnn.cython_compiled_files.im2col_1ch_nchw_cython import im2col_1ch_nchw_cython, col2im_1ch_nchw_cython
from pydtnn.cython_compiled_files.im2col_nchw_cython import im2col_nchw_cython, col2im_nchw_cython
from pydtnn.cython_compiled_files.im2row_1ch_nhwc_cython import im2row_1ch_nhwc_cython, row2im_1ch_nhwc_cython
from pydtnn.cython_compiled_files.im2row_nhwc_cython import im2row_nhwc_cython, row2im_nhwc_cython
from pydtnn.cython_compiled_files.max_pool_2d_nchw_cython import max_pool_2d_fwd_nchw_cython, max_pool_2d_bwd_nchw_cython
from pydtnn.cython_compiled_files.max_pool_2d_nhwc_cython import max_pool_2d_fwd_nhwc_cython, max_pool_2d_bwd_nhwc_cython
from pydtnn.cython_compiled_files.pointwise_conv_cython import pointwise_conv_cython
from pydtnn.cython_compiled_files.relu_cython import relu_cython, capped_relu_cython, leaky_relu_cython
from pydtnn.cython_compiled_files.transpose_cython import \
    transpose_0231_ikj_cython, transpose_0231_ijk_cython, \
    transpose_0312_ijk_cython, transpose_0312_ikj_cython, \
    transpose_1023_jik_cython, transpose_1023_ijk_cython
from pydtnn.cython_compiled_files.adaptive_avg_pooling_nchw_cython import adaptive_avg_pooling_fwd_nchw_cython, adaptive_avg_pooling_bwd_nchw_cython
from pydtnn.cython_compiled_files.adaptive_avg_pooling_nhwc_cython import adaptive_avg_pooling_fwd_nhwc_cython, adaptive_avg_pooling_bwd_nhwc_cython
from pydtnn.cython_compiled_files.memory_cython import memoryview_index
from pydtnn.cython_compiled_files.decimal_cython import round


from pydtnn.cython_compiled_files.sigmoid_cython import sigmoid_fwd_cython, sigmoid_bwd_cython
from pydtnn.cython_compiled_files.log_activation_cython import log_fwd_cython, log_bwd_cython

