from enum import auto, StrEnum

class PYDTNN_TENSOR_FORMAT_ENUM(StrEnum):

    @staticmethod
    def get_num_formats():
        return len(PYDTNN_TENSOR_FORMAT_ENUM)

    # Constants:
    NHWC = auto()
    NCHW = auto()
# --- END PYDTNN_OPS_EVENT_enum --- #

PYDTNN_TENSOR_FORMAT = PYDTNN_TENSOR_FORMAT_ENUM
PYDTNN_TENSOR_FORMATS = PYDTNN_TENSOR_FORMAT.get_num_formats()
# -------------------

def encode_tensor(shape, tensor_format=PYDTNN_TENSOR_FORMAT.NHWC):
    if len(shape) == 3 and tensor_format is PYDTNN_TENSOR_FORMAT.NCHW:
        return shape[2], shape[0], shape[1]
    else:  # Assuming PYDTNN_TENSOR_FORMAT.NHWC
        return shape


def decode_tensor(shape, tensor_format=PYDTNN_TENSOR_FORMAT.NHWC):
    if len(shape) == 3 and tensor_format is PYDTNN_TENSOR_FORMAT.NCHW:
        return shape[1], shape[2], shape[0]
    else:  # Assuming PYDTNN_TENSOR_FORMAT.NHWC
        return shape