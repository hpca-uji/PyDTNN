#
#  This file is part of Python Distributed Training of Neural Networks (PyDTNN)
#
#  Copyright (C) 2021 Universitat Jaume I
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

import ctypes

# noinspection PyUnresolvedReferences
import pycuda.driver as drv


# The below code will allocate the maximum used memory, which will be shared
# among all layers. This code saves having a memory allocation per layer.
ws_size = 1
ws = drv.mem_alloc(ws_size) if ws_size > 0 else 0
ws_ptr = ctypes.c_void_p(int(ws))


def checkConvolutionMemory(size):
    global ws_size
    global ws
    global ws_ptr
    # if a layer requires more memory than the allocated
    # we re-allocated that size
    if size.value > ws_size:
        ws_size = size.value
        ws.free()
        ws = drv.mem_alloc(ws_size) if ws_size > 0 else 0
        ws_ptr = ctypes.c_void_p(int(ws))
