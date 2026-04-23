import numpy as np

from pydtnn.utils.tensor import TensorFormat

def to_cpu(value: np.ndarray, tensor_format) -> np.ndarray:

    match len(value.shape):
        case 1:
            match tensor_format:
                case TensorFormat.NCHW:
                    # shape = (1, *ary.shape, 1, 1)
                    value = np.squeeze(value, axis=(0, 2, 3))
                case TensorFormat.NHWC:
                    # shape = (1, 1, 1, *ary.shape)
                    value = np.squeeze(value, axis=(0, 1, 2))
                case tensor_format:
                    raise NotImplementedError(f"Unsupported tensor format {tensor_format}!")
        case 2:
            match tensor_format:
                case TensorFormat.NCHW:
                    # shape = (*ary.shape, 1, 1)
                    value = np.squeeze(value, axis=(2, 3))
                case TensorFormat.NHWC:
                    # shape = (ary.shape[0], 1, 1, ary.shape[1])
                    value = np.squeeze(value, axis=(1, 2))
                case tensor_format:
                    raise NotImplementedError(f"Unsupported tensor format {tensor_format}!")
        case 3:
            match tensor_format:
                case TensorFormat.NCHW:
                    # shape = (ary.shape[0], 1, ary.shape[1], ary.shape[2])
                    value = np.squeeze(value, axis=(1,))
                case TensorFormat.NHWC:
                    raise NotImplementedError("Shape padding not implemented for 3-dim shape on NHWC")
        case 4:
            # shape = ary.shape
            pass  # exact
        case _:
            raise ValueError(f"The expected len shape are 1, 2, 3 or 4. Shape received: {len(self.ary.shape)}.")

    return value

def from_cpu(value: np.ndarray, tensor_format) -> np.ndarray:
    match len(value.shape):
        case 1:
            match tensor_format:
                case TensorFormat.NCHW:
                    # shape = (1, *ary.shape, 1, 1)
                    value = np.expand_dims(value, axis=(0, 2, 3))
                case TensorFormat.NHWC:
                    # shape = (1, 1, 1, *ary.shape)
                    value = np.expand_dims(value, axis=(0, 1, 2))
                case tensor_format:
                    raise NotImplementedError(f"Unsupported tensor format {tensor_format}!")
        case 2:
            match tensor_format:
                case TensorFormat.NCHW:
                    # shape = (*ary.shape, 1, 1)
                    value = np.expand_dims(value, axis=(2, 3))
                case TensorFormat.NHWC:
                    # shape = (ary.shape[0], 1, 1, ary.shape[1])
                    value = np.expand_dims(value, axis=(1, 2))
                case tensor_format:
                    raise NotImplementedError(f"Unsupported tensor format {tensor_format}!")
        case 3:
            match tensor_format:
                case TensorFormat.NCHW:
                    # shape = (ary.shape[0], 1, ary.shape[1], ary.shape[2])
                    value = np.expand_dims(value, axis=(1,))
                case TensorFormat.NHWC:
                    raise NotImplementedError("Shape padding not implemented for 3-dim shape on NHWC")
        case 4:
            # shape = ary.shape
            pass  # exact
        case _:
            raise ValueError(f"The expected len shape are 1, 2, 3 or 4. Shape received: {len(value.shape)}.")

    return value
