"""
Tracer events
"""
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
    FORWARD = auto()  # Originally: 1
    BACKWARD = auto()  # Originally: 2
    ALLREDUCE_DW = auto()  # Originally: 3
    WAIT_DW = auto()  # Originally: 4
    UPDATE_DW = auto()  # Originally: 5



PYDTNN_MDL_EVENT = 60000001
PYDTNN_MDL_EVENTS = len(PYDTNN_MDL_EVENT_enum)
# ========================== #

# ========================== #
# ==== PYDTNN_OPS_EVENT ==== #
# ========================== #


class PYDTNN_OPS_EVENT_enum(IntEnum):
    OPS_ALLREDUCE_DW = auto()  # Orginally: 1
    BACKWARD_CONVGEMM = auto()  # Orginally: 2
    BACKWARD_CUBLAS_MATMUL_DW = auto()  # Orginally: 3
    BACKWARD_CUBLAS_MATMUL_DX = auto()  # Orginally: 4
    BACKWARD_CUBLAS_MATVEC_DB = auto()  # Orginally: 5
    BACKWARD_CUDNN_DB = auto()  # Orginally: 6
    BACKWARD_CUDNN_DW = auto()  # Orginally: 7
    BACKWARD_CUDNN_DX = auto()  # Orginally: 8
    BACKWARD_DECONV_GEMM = auto()  # Orginally: 9
    BACKWARD_ELTW_SUM = auto()  # Orginally: 10
    BACKWARD_IM2COL = auto()  # Orginally: 11
    BACKWARD_RESHAPE_DW = auto()  # Orginally: 12
    BACKWARD_RESHAPE_DX = auto()  # Orginally: 13
    BACKWARD_SPLIT = auto()  # Orginally: 14
    BACKWARD_SUM_BIASES = auto()  # Orginally: 15
    BACKWARD_TRANSPOSE_DY = auto()  # Orginally: 16
    BACKWARD_TRANSPOSE_W = auto()  # Orginally: 17
    BACKWARD_ADP_AVG_POOL = auto()  # Now: 18
    COMP_DW_MATMUL = auto()  # Orginally: 18
    COMP_DX_COL2IM = auto()  # Orginally: 19
    COMP_DX_MATMUL = auto()  # Orginally: 20
    FORWARD_DEPTHWISE_CONV = auto()  # Orginally: 21
    FORWARD_POINTWISE_CONV = auto()  # Orginally: 22
    FORWARD_CONCAT = auto()  # Orginally: 23
    FORWARD_CONVGEMM = auto()  # Orginally: 24
    FORWARD_CONVWINOGRAD = auto()  # Orginally: 25
    FORWARD_CONVDIRECT = auto()  # Orginally: 26
    FORWARD_CUBLAS_MATMUL = auto()  # Orginally: 27
    FORWARD_CUDNN = auto()  # Orginally: 28
    FORWARD_CUDNN_SUM_BIASES = auto()  # Orginally: 29
    FORWARD_ELTW_SUM = auto()  # Orginally: 30
    FORWARD_IM2COL = auto()  # Orginally: 31
    FORWARD_MATMUL = auto()  # Orginally: 32
    FORWARD_REPLICATE = auto()  # Orginally: 33
    FORWARD_RESHAPE_W = auto()  # Orginally: 34
    FORWARD_RESHAPE_Y = auto()  # Orginally: 35
    FORWARD_SUM_BIASES = auto()  # Orginally: 36
    FORWARD_TRANSPOSE_Y = auto()  # Orginally: 37
    FORWARD_MHA_FC_QKV = auto()  # Orginally: 38
    FORWARD_MHA_MATMUL_QK = auto()  # Orginally: 39
    FORWARD_MHA_SCALARDK = auto()  # Orginally: 40
    FORWARD_MHA_MATMUL_SMV = auto()  # Orginally: 41
    FORWARD_MHA_FC_O = auto()  # Orginally: 42
    BACKWARD_MHA_FC_QKV = auto()  # Orginally: 43
    BACKWARD_MHA_MATMUL_QK = auto()  # Orginally: 44
    BACKWARD_MHA_SCALARDK = auto()  # Orginally: 45
    BACKWARD_MHA_MATMUL_SMV = auto()  # Orginally: 46
    BACKWARD_MHA_FC_O = auto()  # Orginally: 47
    FORWARD_MHA = auto()  # Orginally: 48
    FORWARD_FEEDFORWARD = auto()  # Orginally: 49
    BACKWARD_MHA = auto()  # Orginally: 50
    BACKWARD_FEEDFORWARD = auto()   # Orginally: 51
    FORWARD_ADP_AVG_POOL = auto()  # Orginally: 39


PYDTNN_OPS_EVENT = 60000002
PYDTNN_OPS_EVENTS = len(PYDTNN_OPS_EVENT_enum)
# ========================== #
