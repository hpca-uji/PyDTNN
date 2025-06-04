"""
Tracer events
"""

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
from enum import IntEnum, auto

# ========================== #
# ========= COMMON ========= #
# ========================== #
PYDTNN_EVENT_FINISHED = 0
# ========================== #


# ========================== #
# ==== PYDTNN_MDL_EVENT ==== #
# ========================== #
class PYDTNN_MDL_EVENT_enum(IntEnum):    
    FORWARD      = auto() # Originally: 1
    BACKWARD     = auto() # Originally: 2
    ALLREDUCE_DW = auto() # Originally: 3
    WAIT_DW      = auto() # Originally: 4
    UPDATE_DW    = auto() # Originally: 5

    @staticmethod
    def get_num_events():
        return len(PYDTNN_MDL_EVENT_enum)
# --- END PYDTNN_MDL_EVENT_enum --- #

PYDTNN_MDL_EVENT = 60000001
PYDTNN_MDL_EVENTS = PYDTNN_MDL_EVENT_enum.get_num_events()
# ========================== #

# ========================== #
# ==== PYDTNN_OPS_EVENT ==== #
# ========================== #

class PYDTNN_OPS_EVENT_enum(IntEnum):
    OPS_ALLREDUCE_DW             = auto() # Orginally: 1  || Now: 1
    BACKWARD_CONVGEMM            = auto() # Orginally: 2  || Now: 2
    BACKWARD_CUBLAS_MATMUL_DW    = auto() # Orginally: 3  || Now: 3
    BACKWARD_CUBLAS_MATMUL_DX    = auto() # Orginally: 4  || Now: 4
    BACKWARD_CUBLAS_MATVEC_DB    = auto() # Orginally: 5  || Now: 5
    BACKWARD_CUDNN_DB            = auto() # Orginally: 6  || Now: 6
    BACKWARD_CUDNN_DW            = auto() # Orginally: 7  || Now: 7
    BACKWARD_CUDNN_DX            = auto() # Orginally: 8  || Now: 8
    BACKWARD_DECONV_GEMM         = auto() # Orginally: 9  || Now: 9
    BACKWARD_ELTW_SUM            = auto() # Orginally: 10 || Now: 10
    BACKWARD_IM2COL              = auto() # Orginally: 11 || Now: 11
    BACKWARD_RESHAPE_DW          = auto() # Orginally: 12 || Now: 12
    BACKWARD_RESHAPE_DX          = auto() # Orginally: 13 || Now: 13
    BACKWARD_SPLIT               = auto() # Orginally: 14 || Now: 14
    BACKWARD_SUM_BIASES          = auto() # Orginally: 15 || Now: 15
    BACKWARD_TRANSPOSE_DY        = auto() # Orginally: 16 || Now: 16
    BACKWARD_TRANSPOSE_W         = auto() # Orginally: 17 || Now: 17
    BACKWARD_ADP_AVG_POOL        = auto() # Now: 18
    COMP_DW_MATMUL               = auto() # Orginally: 18 || Now: 19
    COMP_DX_COL2IM               = auto() # Orginally: 19 || Now: 20
    COMP_DX_MATMUL               = auto() # Orginally: 20 || Now: 21
    FORWARD_DEPTHWISE_CONV       = auto() # Orginally: 21 || Now: 22
    FORWARD_POINTWISE_CONV       = auto() # Orginally: 22 || Now: 23
    FORWARD_CONCAT               = auto() # Orginally: 23 || Now: 24
    FORWARD_CONVGEMM             = auto() # Orginally: 24 || Now: 25
    FORWARD_CONVWINOGRAD         = auto() # Orginally: 25 || Now: 26
    FORWARD_CONVDIRECT           = auto() # Orginally: 26 || Now: 27
    FORWARD_CUBLAS_MATMUL        = auto() # Orginally: 27 || Now: 28
    FORWARD_CUDNN                = auto() # Orginally: 28 || Now: 29
    FORWARD_CUDNN_SUM_BIASES     = auto() # Orginally: 29 || Now: 30
    FORWARD_ELTW_SUM             = auto() # Orginally: 30 || Now: 31
    FORWARD_IM2COL               = auto() # Orginally: 31 || Now: 32
    FORWARD_MATMUL               = auto() # Orginally: 32 || Now: 33
    FORWARD_REPLICATE            = auto() # Orginally: 33 || Now: 34
    FORWARD_RESHAPE_W            = auto() # Orginally: 34 || Now: 35
    FORWARD_RESHAPE_Y            = auto() # Orginally: 35 || Now: 36
    FORWARD_SUM_BIASES           = auto() # Orginally: 36 || Now: 37
    FORWARD_TRANSPOSE_Y          = auto() # Orginally: 37 || Now: 38
    FORWARD_ADP_AVG_POOL         = auto() # Now: 39

    @staticmethod
    def get_num_events():
        return len(PYDTNN_OPS_EVENT_enum)
# --- END PYDTNN_OPS_EVENT_enum --- #

PYDTNN_OPS_EVENT = 60000002
PYDTNN_OPS_EVENTS = PYDTNN_OPS_EVENT_enum.get_num_events()
# ========================== #
