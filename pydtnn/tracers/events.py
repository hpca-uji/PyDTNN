"""Tracer events for PyDTNN."""

import logging
from enum import IntEnum, auto

__all__ = (
    "PYDTNN_EVENT_FINISHED",
    "MdlEventEnum",
    "PYDTNN_MDL_EVENT",
    "PYDTNN_MDL_EVENTS",
    "OpsEventEnum",
    "PYDTNN_OPS_EVENT",
    "PYDTNN_OPS_EVENTS",
)

logger = logging.getLogger(__name__)


# ========= COMMON =========
PYDTNN_EVENT_FINISHED = 0


# ==== PYDTNN_MODEL_EVENT ====
class MdlEventEnum(IntEnum):
    """Enumeration of model-level tracer events."""

    FORWARD = auto()
    BACKWARD = auto()
    ALLREDUCE = auto()
    WAITREDUCE = auto()
    OPTIMIZER = auto()


PYDTNN_MDL_EVENT = 60000001
PYDTNN_MDL_EVENTS = len(MdlEventEnum)

# ==== PYDTNN_OPERATION_EVENT ====


class OpsEventEnum(IntEnum):
    """Enumeration of operation-level tracer events."""

    ALLREDUCE = auto()
    BACKWARD_CONVGEMM = auto()
    BACKWARD_CUBLAS_MATMUL_DW = auto()
    BACKWARD_CUBLAS_MATMUL_DX = auto()
    BACKWARD_CUBLAS_MATVEC_DB = auto()
    BACKWARD_CUDNN_DB = auto()
    BACKWARD_CUDNN_DW = auto()
    BACKWARD_CUDNN_DX = auto()
    BACKWARD_DECONV_GEMM = auto()
    BACKWARD_ELTW_SUM = auto()
    BACKWARD_IM2COL = auto()
    BACKWARD_RESHAPE_DW = auto()
    BACKWARD_RESHAPE_DX = auto()
    BACKWARD_SPLIT = auto()
    BACKWARD_SUM_BIASES = auto()
    BACKWARD_TRANSPOSE_DY = auto()
    BACKWARD_TRANSPOSE_W = auto()
    BACKWARD_ADP_AVG_POOL = auto()  # Now: 18
    COMP_DW_MATMUL = auto()
    COMP_DX_COL2IM = auto()
    COMP_DX_MATMUL = auto()
    FORWARD_DEPTHWISE_CONV = auto()
    FORWARD_POINTWISE_CONV = auto()
    FORWARD_CONCAT = auto()
    FORWARD_CONVGEMM = auto()
    FORWARD_CONVWINOGRAD = auto()
    FORWARD_CONVDIRECT = auto()
    FORWARD_CUBLAS_MATMUL = auto()
    FORWARD_CUDNN = auto()
    FORWARD_CUDNN_SUM_BIASES = auto()
    FORWARD_ELTW_SUM = auto()
    FORWARD_IM2COL = auto()
    FORWARD_MATMUL = auto()
    FORWARD_REPLICATE = auto()
    FORWARD_RESHAPE_W = auto()
    FORWARD_RESHAPE_Y = auto()
    FORWARD_SUM_BIASES = auto()
    FORWARD_TRANSPOSE_Y = auto()
    FORWARD_MHA_FC_QKV = auto()
    FORWARD_MHA_MATMUL_QK = auto()
    FORWARD_MHA_SCALARDK = auto()
    FORWARD_MHA_MATMUL_SMV = auto()
    FORWARD_MHA_FC_O = auto()
    BACKWARD_MHA_FC_QKV = auto()
    BACKWARD_MHA_MATMUL_QK = auto()
    BACKWARD_MHA_SCALARDK = auto()
    BACKWARD_MHA_MATMUL_SMV = auto()
    BACKWARD_MHA_FC_O = auto()
    FORWARD_MHA = auto()
    FORWARD_FEEDFORWARD = auto()
    BACKWARD_MHA = auto()
    BACKWARD_FEEDFORWARD = auto()
    FORWARD_ADP_AVG_POOL = auto()
    LAYER_ENCODE = auto()
    LAYER_DECODE = auto()


PYDTNN_OPS_EVENT = 60000002
PYDTNN_OPS_EVENTS = len(OpsEventEnum)
