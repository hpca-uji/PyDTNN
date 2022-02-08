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

from .conv2d_conv_gemm import Conv2DConvGemmTestCase
from .conv2d_conv_gemm_slow import Conv2DConvGemmSlowTestCase
from .conv_gemm import ConvGemmTestCase
from .check_conv_gemm_models import CheckConvGemmModels
from .check_conv_gemm_nchw_models import CheckConvGemmNCHWModels
from .check_tensor_format_models import CheckTensorFormatModels
try:
    from .check_gpu_models import CheckGPUModels
except (ModuleNotFoundError, ImportError):
    pass
