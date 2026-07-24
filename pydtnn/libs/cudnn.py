"""Python interface to the NVIDIA cuDNN library"""

# Source: https://github.com/hannes-brt/cudnn-python-wrappers

import ctypes
import logging
import sys

__all__ = (
    "CudnnConvolutionBwdDataAlgoPerf",
    "CudnnConvolutionBwdFilterAlgoPerf",
    "CudnnConvolutionFwdAlgoPerf",
    "CudnnError",
    "cudnnActivationBackward",
    "cudnnActivationForward",
    "cudnnAddTensor",
    "cudnnBatchNormalizationBackward",
    "cudnnBatchNormalizationForwardInference",
    "cudnnBatchNormalizationForwardTraining",
    "cudnnCheckStatus",
    "cudnnConvolutionBackwardBias",
    "cudnnConvolutionBackwardData",
    "cudnnConvolutionBackwardFilter",
    "cudnnConvolutionForward",
    "cudnnCreate",
    "cudnnCreateActivationDescriptor",
    "cudnnCreateAttnDescriptor",
    "cudnnCreateConvolutionDescriptor",
    "cudnnCreateDropoutDescriptor",
    "cudnnCreateFilterDescriptor",
    "cudnnCreatePoolingDescriptor",
    "cudnnCreateSeqDataDescriptor",
    "cudnnCreateTensorDescriptor",
    "cudnnDeriveBNTensorDescriptor",
    "cudnnDestroy",
    "cudnnDestroyAttnDescriptor",
    "cudnnDestroyConvolutionDescriptor",
    "cudnnDestroyFilterDescriptor",
    "cudnnDestroyPoolingDescriptor",
    "cudnnDestroySeqDataDescriptor",
    "cudnnDestroyTensorDescriptor",
    "cudnnDropoutBackward",
    "cudnnDropoutForward",
    "cudnnDropoutGetReserveSpaceSize",
    "cudnnDropoutGetStatesSize",
    "cudnnFindConvolutionBackwardDataAlgorithm",
    "cudnnFindConvolutionBackwardFilterAlgorithm",
    "cudnnFindConvolutionForwardAlgorithm",
    "cudnnGetConvolution2dDescriptor",
    "cudnnGetConvolution2dForwardOutputDim",
    "cudnnGetConvolutionBackwardDataWorkspaceSize",
    "cudnnGetConvolutionBackwardFilterWorkspaceSize",
    "cudnnGetConvolutionForwardWorkspaceSize",
    "cudnnGetFilter4dDescriptor",
    "cudnnGetMultiHeadAttnBuffers",
    "cudnnGetMultiHeadAttnWeights",
    "cudnnGetNormalizationBackwardWorkspaceSize",
    "cudnnGetNormalizationForwardTrainingWorkspaceSize",
    "cudnnGetNormalizationTrainingReserveSpaceSize",
    "cudnnGetPooling2dDescriptor",
    "cudnnGetPooling2dForwardOutputDim",
    "cudnnGetStream",
    "cudnnGetTensor4dDescriptor",
    "cudnnGetVersion",
    "cudnnMultiHeadAttnBackwardData",
    "cudnnMultiHeadAttnBackwardWeights",
    "cudnnMultiHeadAttnForward",
    "cudnnNormalizationBackward",
    "cudnnNormalizationForwardInference",
    "cudnnNormalizationForwardTraining",
    "cudnnPoolingBackward",
    "cudnnPoolingForward",
    "cudnnScaleTensor",
    "cudnnSetActivationDescriptor",
    "cudnnSetAttnDescriptor",
    "cudnnSetConvolution2dDescriptor",
    "cudnnSetConvolutionGroupCount",
    "cudnnSetConvolutionMathType",
    "cudnnSetConvolutionNdDescriptor",
    "cudnnSetDropoutDescriptor",
    "cudnnSetFilter4dDescriptor",
    "cudnnSetPooling2dDescriptor",
    "cudnnSetSeqDataDescriptor",
    "cudnnSetStream",
    "cudnnSetTensor",
    "cudnnSetTensor4dDescriptor",
    "cudnnSetTensor4dDescriptorEx",
    "cudnnSoftmaxBackward",
    "cudnnSoftmaxForward",
    "cudnnTransformTensor",
)

logger = logging.getLogger(__name__)


if sys.platform in ("linux2", "linux"):
    _libcudnn_libname_list = ["libcudnn.so", "libcudnn.so.7", "libcudnn.so.6.0.21"]
elif sys.platform == "darwin":
    _libcudnn_libname_list = ["libcudnn.dylib", "libcudnn.6.dylib"]
elif sys.platform == "win32":
    _libcudnn_libname_list = ["cudnn64_6.dll"]
else:
    raise NotImplementedError("PyDTNN CUDNN: current platform is not yet supported!")

_libcudnn = None
for _libcudnn_libname in _libcudnn_libname_list:
    try:
        _libcudnn = ctypes.cdll.LoadLibrary(_libcudnn_libname)
    except OSError:
        pass
    else:
        break
if _libcudnn is None:
    raise OSError("cuDNN library not found")

# cuDNN error
_libcudnn.cudnnGetErrorString.restype = ctypes.c_char_p
_libcudnn.cudnnGetErrorString.argtypes = [ctypes.c_int]


class CudnnError(Exception):
    """Exception raised for errors in the cuDNN library."""

    def __init__(self, status: int) -> None:
        """Inizialzie exception"""
        self.status = status

    def __str__(self) -> str:
        """Return the cuDNN error string."""
        assert _libcudnn
        error = _libcudnn.cudnnGetErrorString(self.status)
        return f"{error}"


# Data layout specification
# cudnnTensorFormat_t is an enumerated type used by
# cudnnSetTensor4dDescriptor() to create a tensor with a pre-defined layout.
type CudnnTensorFormat = dict[str, int]
cudnnTensorFormat = {
    "CUDNN_TENSOR_NCHW": 0,  # This tensor format specifies that the data
    # is laid out in the following order: image,
    # features map, rows, columns. The strides
    # are implicitly defined in such a way that
    # the data are contiguous in memory with no
    # padding between images, feature maps,
    # rows, and columns; the columns are the
    # inner dimension and the images are the
    # outermost dimension.
    "CUDNN_TENSOR_NHWC": 1,  # This tensor format specifies that the data
    # is laid out in the following order: image,
    # rows, columns, features maps. The strides
    # are implicitly defined in such a way that
    # the data are contiguous in memory with no
    # padding between images, rows, columns, and
    # features maps; the feature maps are the
    # inner dimension and the images are the
    # outermost dimension.
    "CUDNN_TENSOR_NCHW_VECT_C": 2,  # This tensor format specifies that the data
    # is laid out in the following order: batch
    # size, feature maps, rows, columns. However,
    # each element of the tensor is a vector of
    # multiple feature maps. The length of the
    # vector is carried by the data type of the
    # tensor. The strides are implicitly defined
    # in such a way that the data are contiguous
    # in memory with no padding between images,
    # feature maps, rows, and columns; the
    # columns are the inner dimension and the
    # images are the outermost dimension. This
    # format is only supported with tensor data
    # type CUDNN_DATA_INT8x4.
}

# Data type
# cudnnDataType_t is an enumerated type indicating the data type to which a tensor
# descriptor or filter descriptor refers.
type CudnnDataType = dict[str, int]
cudnnDataType = {
    "CUDNN_DATA_FLOAT": 0,  # The data is 32-bit single-precision floating point
    # ( float ).
    "CUDNN_DATA_DOUBLE": 1,  # The data is 64-bit double-precision floating point
    # ( double ).
    "CUDNN_DATA_HALF": 2,  # The data is 16-bit half-precision floating point
    # ( half ).
    "CUDNN_DATA_INT8": 3,  # The data is 8-bit signed integer.
    "CUDNN_DATA_INT32": 4,  # The data is 32-bit signed integer.
    "CUDNN_DATA_INT8x4": 5,  # The data is 32-bit element composed of 4 8-bit
    # signed integer. This data type is only supported
    # with tensor tensor_format CUDNN_TENSOR_NCHW_VECT_C.
}

# Math type
# cudnnMathType_t is an enumerated type used to indicate if the use of Tensor Core
# operations is permitted in a given library routine.
type CudnnMathType = dict[str, int]
cudnnMathType = {
    "CUDNN_DEFAULT_MATH": 0,  # Tensor Core operations are not used on
    # pre-NVIDIA A100 GPU devices. On A100 GPU architecture devices,
    # Tensor Core TF32 operation is permitted.
    "CUDNN_TENSOR_OP_MATH": 1,  # The use of Tensor Core operations is permitted
    # but will not actively perform datatype down conversion on tensors in order
    # to utilize Tensor Cores.
    "CUDNN_TENSOR_OP_MATH_ALLOW_CONVERSION": 2,  # The use of Tensor Core operations
    # is permitted and will actively perform datatype down conversion on tensors
    # in order to utilize Tensor Cores.
    "CUDNN_FMA_MATH": 3,  # Restricted to only kernels that use FMA instructions.
}


# cudnnSeqDataAxis_t is an enumerated type used by cudnnSetSeqDataDescriptor()
# type cudnnSeqDataAxis = dict[str, int]
cudnnSeqDataAxis = {
    "CUDNN_SEQDATA_TIME_DIM": 0,  # Identifies the TIME (sequence length) dimension or
    #  specifies the TIME in the data layout.
    "CUDNN_SEQDATA_BATCH_DIM": 1,  # Identifies the BATCH dimension or specifies the BATCH
    # in the data layout.
    "CUDNN_SEQDATA_BEAM_DIM": 2,  # Identifies the BEAM dimension or specifies the BEAM in
    # the data layout.
    "CUDNN_SEQDATA_VECT_DIM": 3,  # Identifies the VECT (vector) dimension or specifies the
    # VECT in the data layout.
}

# type cudnnMultiHeadAttnWeightKind = dict[str, int]
cudnnMultiHeadAttnWeightKind = {
    "CUDNN_MH_ATTN_Q_WEIGHTS": 0,  # Selects the input projection weights for queries.
    "CUDNN_MH_ATTN_K_WEIGHTS": 1,  # Selects the input projection weights for keys.
    "CUDNN_MH_ATTN_V_WEIGHTS": 2,  # Selects the input projection weights for values.
    "CUDNN_MH_ATTN_O_WEIGHTS": 3,  # Selects the output projection weights.
    "CUDNN_MH_ATTN_Q_BIASES": 4,  # Selects the input projection biases for queries.
    "CUDNN_MH_ATTN_K_BIASES": 5,  # Selects the input projection biases for keys.
    "CUDNN_MH_ATTN_V_BIASES": 6,  # Selects the input projection biases for values.
    "CUDNN_MH_ATTN_O_BIASES": 7,  # Selects the output projection biases.
}


# type cudnnAttnMode = dict[str, int]
cudnnAttnMode = {
    # Forward declaration of mapping between Q and K , V vectors when the beam
    # size is greater than one in the Q input. Multiple Q vectors from the
    # same beam bundle map to the same K , V vectors. This means that beam
    # sizes in the K , V sets are equal to one.
    "CUDNN_ATTN_QUERYMAP_ALL_TO_ONE": 0,
    # Forward declaration of mapping between Q and K , V vectors when the beam
    # size is greater than one in the Q input. Multiple Q vectors from the
    # same beam bundle map to different K , V vectors. This requires beam
    # sizes in K , V sets to be the same as in the Q input.
    "CUDNN_ATTN_QUERYMAP_ONE_TO_ONE": 1,
    # Use no biases in the attention input and output projections.
    "CUDNN_ATTN_DISABLE_PROJ_BIASES": 0,
    # Use extra biases in the attention input and output projections. In this
    # case the projected K ¯ vectors are computed as K i ¯ = W K , i K + b * 1
    # , 1 , ..., 1 1 × n , where n is the number of columns in the K matrix.
    # In other words, the same column vector b is added to all columns of K
    # after the weight matrix multiplication.
    "CUDNN_ATTN_ENABLE_PROJ_BIASES": 2,
}


# type cudnnWgradMode = dict[str, int]
cudnnWgradMode = {
    # A weight gradient component corresponding to a new batch of inputs is
    # added to previously evaluated weight gradients. Before using this mode,
    # the buffer holding weight gradients should be initialized to zero.
    # Alternatively, the first API call outputting to an uninitialized buffer
    # should use the CUDNN_WGRAD_MODE_SET option.
    "CUDNN_WGRAD_MODE_ADD": 0,
    # A weight gradient component, corresponding to a new batch of inputs,
    # overwrites previously stored weight gradients in the output buffer.
    "CUDNN_WGRAD_MODE_SET": 1,
}


# cudnnAddMode_t is an enumerated type used by cudnnAddTensor() to specify how
# a bias tensor is added to an input/output tensor.
type CudnnAddMode = dict[str, int]
cudnnAddMode = {
    "CUDNN_ADD_IMAGE": 0,
    "CUDNN_ADD_SAME_HW": 0,  # In this mode, the bias tensor is defined as one
    # image with one feature map. This image will be
    # added to every feature map of every image of the
    # input/output tensor.
    "CUDNN_ADD_FEATURE_MAP": 1,
    "CUDNN_ADD_SAME_CHW": 1,  # In this mode, the bias tensor is defined as one
    # image with multiple feature maps. This image
    # will be added to every image of the input/output
    # tensor.
    "CUDNN_ADD_SAME_C": 2,  # In this mode, the bias tensor is defined as one
    # image with multiple feature maps of dimension
    # 1x1; it can be seen as an vector of feature maps.
    # Each feature map of the bias tensor will be added
    # to the corresponding feature map of all height-by-
    # width pixels of every image of the input/output
    # tensor.
    "CUDNN_ADD_FULL_TENSOR": 3,  # In this mode, the bias tensor has the same
    # dimensions as the input/output tensor. It will be
    # added point-wise to the input/output tensor.
}

# cudnnConvolutionMode_t is an enumerated type used by
# cudnnSetConvolutionDescriptor() to configure a convolution descriptor. The
# filter used for the convolution can be applied in two different ways, corresponding
# mathematically to a convolution or to a cross-correlation. (A cross-correlation is
# equivalent to a convolution with its filter rotated by 180 degrees.)
type CudnnConvolutionMode = dict[str, int]
cudnnConvolutionMode = {
    "CUDNN_CONVOLUTION": 0,  # In this mode, a convolution operation will be done
    # when applying the filter to the images.
    "CUDNN_CROSS_CORRELATION": 1,  # In this mode, a cross-correlation operation will
    # be done when applying the filter to the images.
}

# cudnnConvolutionFwdPreference_t is an enumerated type used by
# cudnnGetConvolutionForwardAlgorithm() to help the choice of the algorithm used for the
# forward convolution.
type CudnnConvolutionFwdPreference = dict[str, int]
cudnnConvolutionFwdPreference = {
    "CUDNN_CONVOLUTION_FWD_NO_WORKSPACE": 0,  # In this configuration, the routine
    # cudnnGetConvolutionForwardAlgorithm() is guaranteed to return
    # an algorithm that does not require any extra workspace to be
    # provided by the user.
    "CUDNN_CONVOLUTION_FWD_PREFER_FASTEST": 1,  # In this configuration, the routine
    # cudnnGetConvolutionForwardAlgorithm() will return the fastest
    # algorithm regardless how much workspace is needed to execute it.
    "CUDNN_CONVOLUTION_FWD_SPECIFY_WORKSPACE_LIMIT": 2,  # In this configuration, the routine
    # cudnnGetConvolutionForwardAlgorithm() will return the fastest
    # algorithm that fits within the memory limit that the user provided.
}

# cudnnConvolutionFwdAlgo_t is an enumerated type that exposes the different algorithm
# available to execute the forward convolution operation.
type CudnnConvolutionFwdAlgo = dict[str, int]
cudnnConvolutionFwdAlgo = {
    "CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM": 0,  # This algorithm expresses the convolution
    # as a matrix product without actually explicitly forming the matrix
    # that holds the input tensor data.
    "CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM": (
        1
    ),  # This algorithm expresses the convolution
    # as a matrix product without actually explicitly forming the matrix
    # that holds the input tensor data, but still needs some memory
    # workspace to precompute some indices in order to facilitate the
    # implicit construction of the matrix that holds the input tensor data.
    "CUDNN_CONVOLUTION_FWD_ALGO_GEMM": 2,  # This algorithm expresses the convolution as an
    # explicit matrix product. A significant memory workspace is needed to
    # store the matrix that holds the input tensor data.
    "CUDNN_CONVOLUTION_FWD_ALGO_DIRECT": 3,  # This algorithm expresses the convolution as a
    # direct convolution (e.g without implicitly or explicitly doing a
    # matrix multiplication).
    "CUDNN_CONVOLUTION_FWD_ALGO_FFT": 4,
    "CUDNN_CONVOLUTION_FWD_ALGO_FFT_TILING": 5,
    "CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD": 6,
    "CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD_NONFUSED": 7,
    "CUDNN_CONVOLUTION_FWD_ALGO_COUNT": 8,
}

type CudnnConvolutionBwdDataPreference = dict[str, int]
cudnnConvolutionBwdDataPreference = {
    "CUDNN_CONVOLUTION_BWD_DATA_NO_WORKSPACE": 0,
    "CUDNN_CONVOLUTION_BWD_DATA_PREFER_FASTEST": 1,
    "CUDNN_CONVOLUTION_BWD_DATA_SPECIFY_WORKSPACE_LIMIT": 2,
}

type CudnnConvolutionBwdDataAlgo = dict[str, int]
cudnnConvolutionBwdDataAlgo = {
    "CUDNN_CONVOLUTION_BWD_DATA_ALGO_0": 0,
    "CUDNN_CONVOLUTION_BWD_DATA_ALGO_1": 1,
    "CUDNN_CONVOLUTION_BWD_DATA_ALGO_FFT": 2,
    "CUDNN_CONVOLUTION_BWD_DATA_ALGO_FFT_TILING": 3,
    "CUDNN_CONVOLUTION_BWD_DATA_ALGO_WINOGRAD": 4,
    "CUDNN_CONVOLUTION_BWD_DATA_ALGO_WINOGRAD_NONFUSED": 5,
    "CUDNN_CONVOLUTION_BWD_DATA_ALGO_COUNT": 6,
}

type CudnnConvolutionBwdFilterPreference = dict[str, int]
cudnnConvolutionBwdFilterPreference = {
    "CUDNN_CONVOLUTION_BWD_FILTER_NO_WORKSPACE": 0,
    "CUDNN_CONVOLUTION_BWD_FILTER_PREFER_FASTEST": 1,
    "CUDNN_CONVOLUTION_BWD_FILTER_SPECIFY_WORKSPACE_LIMIT": 2,
}

type CudnnConvolutionBwdFilterAlgo = dict[str, int]
cudnnConvolutionBwdFilterAlgo = {
    "CUDNN_CONVOLUTION_BWD_FILTER_ALGO_0": 0,
    "CUDNN_CONVOLUTION_BWD_FILTER_ALGO_1": 1,
    "CUDNN_CONVOLUTION_BWD_FILTER_ALGO_FFT": 2,
    "CUDNN_CONVOLUTION_BWD_FILTER_ALGO_3": 3,
    "CUDNN_CONVOLUTION_BWD_FILTER_ALGO_WINOGRAD": 4,
    "CUDNN_CONVOLUTION_BWD_FILTER_ALGO_WINOGRAD_NONFUSED": 5,
    "CUDNN_CONVOLUTION_BWD_FILTER_ALGO_FFT_TILING": 6,
    "CUDNN_CONVOLUTION_BWD_FILTER_ALGO_COUNT": 7,
}

type CudnnBatchNormMode = dict[str, int]
cudnnBatchNormMode = {
    "CUDNN_BATCHNORM_PER_ACTIVATION": 0,
    "CUDNN_BATCHNORM_SPATIAL": 1,
    "CUDNN_BATCHNORM_SPATIAL_PERSISTENT": 2,
}

# cudnnSoftmaxAlgorithm_t is used to select an implementation of the softmax
# function used in cudnnSoftmaxForward() and cudnnSoftmaxBackward().
type CudnnSoftmaxAlgorithm = dict[str, int]
cudnnSoftmaxAlgorithm = {
    "CUDNN_SOFTMAX_FAST": 0,  # This implementation applies the straightforward
    # softmax operation.
    "CUDNN_SOFTMAX_ACCURATE": 1,  # This implementation applies a scaling to the input
    # to avoid any potential overflow.
    "CUDNN_SOFTMAX_LOG": 2,  # This implementation applied the Log
    # softmax operation, scaling the input to avoid any potential
    # overflow.
}

# cudnnSoftmaxMode_t is used to select over which data the cudnnSoftmaxForward()
# and cudnnSoftmaxBackward() are computing their results.
type CudnnSoftmaxMode = dict[str, int]
cudnnSoftmaxMode = {
    "CUDNN_SOFTMAX_MODE_INSTANCE": 0,  # The softmax operation is computed per image (N)
    # across the dimensions C,H,W.
    "CUDNN_SOFTMAX_MODE_CHANNEL": 1,  # The softmax operation is computed per spatial
    # location (H,W) per image (N) across the dimension
    # C.
}

# cudnnPoolingMode_t is an enumerated type passed to
# cudnnSetPoolingDescriptor() to select the pooling method to be used by
# cudnnPoolingForward() and cudnnPoolingBackward() .
type CudnnPoolingMode = dict[str, int]
cudnnPoolingMode = {
    "CUDNN_POOLING_MAX": 0,  # The maximum value inside the pooling window will
    # be used.
    "CUDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING": 1,  # The values inside the
    # pooling window will be averaged and this count
    # includes padded values.
    "CUDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING": 2,  # The values inside the
    #  pooling window will be averaged and this count
    # does not include padded values.
    "CUDNN_POOLING_MAX_DETERMINISTIC": 3,  # The maximum value inside the pooling
    # window is used. The algorithm used is
    # deterministic.
}
# cudnnNanPropagation_t is an enumerated type used to indicate if a given routine
# should propagate Nan numbers. This enumerated type is used as a field for the
# cudnnActivationDescriptor_t descriptor and cudnnPoolingDescriptor_t descriptor
type CudnnNanPropagation = dict[str, int]
cudnnNanPropagation = {"CUDNN_NOT_PROPAGATE_NAN": 0, "CUDNN_PROPAGATE_NAN": 1}
# cudnnActivationMode_t is an enumerated type used to select the neuron activation
# function used in cudnnActivationForward() and cudnnActivationBackward() .
type CudnnActivationMode = dict[str, int]
cudnnActivationMode = {
    "CUDNN_ACTIVATION_SIGMOID": 0,  # sigmoid function
    "CUDNN_ACTIVATION_RELU": 1,  # rectified linear function
    "CUDNN_ACTIVATION_TANH": 2,  # hyperbolic tangent function
    "CUDNN_ACTIVATION_CLIPPED_RELU": 3,
    "CUDNN_ACTIVATION_ELU": 4,
    "CUDNN_ACTIVATION_IDENTITY": 5,
}


def cudnnCheckStatus(status: int) -> None:
    """Raise cuDNN exception

    Raise an exception corresponding to the specified cuDNN error code.

    Parameters
    ----------
    status : int
        cuDNN error code
    """

    if status != 0:
        raise CudnnError(status)


# Helper functions

_libcudnn.cudnnGetVersion.restype = ctypes.c_size_t
_libcudnn.cudnnGetVersion.argtypes = []


def cudnnGetVersion() -> int:
    """Get cuDNN Version."""
    assert _libcudnn
    return _libcudnn.cudnnGetVersion()


_libcudnn.cudnnCreate.restype = int
_libcudnn.cudnnCreate.argtypes = [ctypes.c_void_p]


def cudnnCreate() -> int:
    """Initialize cuDNN.

    Initializes cuDNN and returns a handle to the cuDNN context.

    Returns
    -------
    handle : cudnnHandle
        cuDNN context
    """

    handle = ctypes.c_void_p()
    assert _libcudnn
    status = _libcudnn.cudnnCreate(ctypes.byref(handle))
    cudnnCheckStatus(status)
    value = handle.value
    assert value
    return value


_libcudnn.cudnnDestroy.restype = int
_libcudnn.cudnnDestroy.argtypes = [ctypes.c_void_p]


def cudnnDestroy(handle: int) -> None:
    """Release cuDNN resources.

    Release hardware resources used by cuDNN.

    Parameters
    ----------
    handle : cudnnHandle
        cuDNN context.
    """

    assert _libcudnn
    status = _libcudnn.cudnnDestroy(ctypes.c_void_p(handle))
    cudnnCheckStatus(status)


_libcudnn.cudnnSetStream.restype = int
_libcudnn.cudnnSetStream.argtypes = [ctypes.c_void_p, ctypes.c_void_p]


def cudnnSetStream(handle: int, stream_id: int) -> None:
    """Set current cuDNN library stream.

    Parameters
    ----------
    handle : cudnnHandle
        cuDNN context.
    stream_id : cudaStream
        Stream Id.
    """

    assert _libcudnn
    status = _libcudnn.cudnnSetStream(handle, stream_id)
    cudnnCheckStatus(status)


_libcudnn.cudnnGetStream.restype = int
_libcudnn.cudnnGetStream.argtypes = [ctypes.c_void_p, ctypes.c_void_p]


def cudnnGetStream(handle: int) -> int:
    """Get current cuDNN library stream.

    Parameters
    ----------
    handle : int
        cuDNN context.
    Returns
    -------
    stream_id : int
        Stream ID.
    """

    stream_id = ctypes.c_void_p()
    assert _libcudnn
    status = _libcudnn.cudnnGetStream(handle, ctypes.byref(stream_id))
    cudnnCheckStatus(status)
    value = stream_id.value
    assert value is not None
    return value


_libcudnn.cudnnCreateActivationDescriptor.restype = int
_libcudnn.cudnnCreateActivationDescriptor.argtypes = [ctypes.c_void_p]


def cudnnCreateActivationDescriptor() -> int:
    """Create a Activation descriptor object.

    Allocates a cudnnActivationDescriptor_t structure and returns a pointer to it.

    Returns
    -------
    Activation_descriptor : int
        Tensor descriptor.
    """

    activation = ctypes.c_void_p()
    assert _libcudnn
    status = _libcudnn.cudnnCreateActivationDescriptor(ctypes.byref(activation))
    cudnnCheckStatus(status)
    value = activation.value
    assert value
    return value


_libcudnn.cudnnSetActivationDescriptor.restype = int
_libcudnn.cudnnSetActivationDescriptor.argtypes = [
    ctypes.c_void_p,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_double,
]


def cudnnSetActivationDescriptor(activation_desc: int, mode: int, nan: int, coef: float) -> None:
    """Set a Activation descriptor object.

    Allocates a cudnnActivationDescriptor_t structure and returns a pointer to it.

    Parameters
    -----------
    activation_desc : cudnnActivationDescriptor
        Handle to a previously created activation descriptor.
    mode : cudnnActivationMode
        Input. Enumerant to specify the activation mode.
    nan : cudnnNanPropagation
        Enumerate to specify the nan propagation
    coef : double
        Input. Floating point number. When the activation mode
        (refer to cudnnActivationMode) is set to CUDNN_ACTIVATION_CLIPPED_RELU,
        this input specifies the clipping threshold;
        and when the activation mode is set to CUDNN_ACTIVATION_RELU,
        this input specifies the upper bound.
    """

    assert _libcudnn
    status = _libcudnn.cudnnSetActivationDescriptor(activation_desc, mode, nan, coef)
    cudnnCheckStatus(status)


_libcudnn.cudnnCreateTensorDescriptor.restype = int
_libcudnn.cudnnCreateTensorDescriptor.argtypes = [ctypes.c_void_p]


def cudnnCreateTensorDescriptor() -> int:
    """Create a Tensor descriptor object.

    Allocates a cudnnTensorDescriptor_t structure and returns a pointer to it.

    Returns
    -------
    tensor_descriptor : int
        Tensor descriptor.
    """

    tensor = ctypes.c_void_p()
    assert _libcudnn
    status = _libcudnn.cudnnCreateTensorDescriptor(ctypes.byref(tensor))
    cudnnCheckStatus(status)
    value = tensor.value
    assert value
    return value


_libcudnn.cudnnSetTensor4dDescriptor.restype = int
_libcudnn.cudnnSetTensor4dDescriptor.argtypes = [
    ctypes.c_void_p,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
]


def cudnnSetTensor4dDescriptor(
    tensor_desc: int, tensor_format: int, data_type: int, n: int, c: int, h: int, w: int
) -> None:
    """Initialize a previously created Tensor 4D object.

    This function initializes a previously created Tensor4D descriptor object. The strides of
    the four dimensions are inferred from the tensor_format parameter and set in such a way that
    the data is contiguous in memory with no padding between dimensions.

    Parameters
    ----------
    tensor_desc : cudnnTensorDescriptor
        Handle to a previously created tensor descriptor.
    tensor_format : cudnnTensorFormat
        Type of tensor_format.
    data_type : cudnnDataType
        Data type.
    n : int
        Number of images.
    c : int
        Number of feature maps per image.
    h : int
        Height of each feature map.
    w : int
        Width of each feature map.
    """

    assert _libcudnn
    status = _libcudnn.cudnnSetTensor4dDescriptor(tensor_desc, tensor_format, data_type, n, c, h, w)
    cudnnCheckStatus(status)


_libcudnn.cudnnSetTensor4dDescriptorEx.restype = int
_libcudnn.cudnnSetTensor4dDescriptorEx.argtypes = [
    ctypes.c_void_p,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
]


def cudnnSetTensor4dDescriptorEx(
    tensor_desc: int,
    data_type: int,
    n: int,
    c: int,
    h: int,
    w: int,
    n_stride: int,
    c_stride: int,
    h_stride: int,
    w_stride: int,
) -> None:
    """Initialize a Tensor descriptor object with strides.

    This function initializes a previously created generic Tensor descriptor object into a
    4D tensor, similarly to cudnnSetTensor4dDescriptor but with the strides explicitly
    passed as parameters. This can be used to lay out the 4D tensor in any order or simply to
    define gaps between dimensions.

    Parameters
    ----------
    tensor_desc : cudnnTensorDescriptor_t
        Handle to a previously created tensor descriptor.
    data_type : cudnnDataType
        Data type.
    n : int
        Number of images.
    c : int
        Number of feature maps per image.
    h : int
        Height of each feature map.
    w : int
        Width of each feature map.
    n_stride : int
        Stride between two consecutive images.
    c_stride : int
        Stride between two consecutive feature maps.
    h_stride : int
        Stride between two consecutive rows.
    w_stride : int
        Stride between two consecutive columns.
    """

    assert _libcudnn
    status = _libcudnn.cudnnSetTensor4dDescriptorEx(
        tensor_desc, data_type, n, c, h, w, n_stride, c_stride, h_stride, w_stride
    )
    cudnnCheckStatus(status)


_libcudnn.cudnnGetTensor4dDescriptor.restype = int
_libcudnn.cudnnGetTensor4dDescriptor.argtypes = [
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_void_p,
]


def cudnnGetTensor4dDescriptor(
    tensor_desc: int,
) -> tuple[int, int, int, int, int, int, int, int, int]:
    """Get parameters of a Tensor descriptor object.

    This function queries the parameters of the previously initialized Tensor4D descriptor
    object.

    Parameters
    ----------
    tensor_desc : cudnnTensorDescriptor
        Handle to a previously initialized tensor descriptor.
    Returns
    -------
    data_type : cudnnDataType
        Data type.
    n : int
        Number of images.
    c : int
        Number of feature maps per image.
    h : int
        Height of each feature map.
    w : int
        Width of each feature map.
    n_stride : int
        Stride between two consecutive images.
    c_stride : int
        Stride between two consecutive feature maps.
    h_stride : int
        Stride between two consecutive rows.
    w_stride : int
        Stride between two consecutive columns.
    """

    data_type = ctypes.c_int()
    n = ctypes.c_int()
    c = ctypes.c_int()
    h = ctypes.c_int()
    w = ctypes.c_int()
    n_stride = ctypes.c_int()
    c_stride = ctypes.c_int()
    h_stride = ctypes.c_int()
    w_stride = ctypes.c_int()

    assert _libcudnn
    status = _libcudnn.cudnnGetTensor4dDescriptor(
        tensor_desc,
        ctypes.byref(data_type),
        ctypes.byref(n),
        ctypes.byref(c),
        ctypes.byref(h),
        ctypes.byref(w),
        ctypes.byref(n_stride),
        ctypes.byref(c_stride),
        ctypes.byref(h_stride),
        ctypes.byref(w_stride),
    )
    cudnnCheckStatus(status)

    return (
        data_type.value,
        n.value,
        c.value,
        h.value,
        w.value,
        n_stride.value,
        c_stride.value,
        h_stride.value,
        w_stride.value,
    )


_libcudnn.cudnnDestroyTensorDescriptor.restype = int
_libcudnn.cudnnDestroyTensorDescriptor.argtypes = [ctypes.c_void_p]


def cudnnDestroyTensorDescriptor(tensor_desc: int) -> None:
    """Destroy a Tensor descriptor.

    This function destroys a previously created Tensor descriptor object.

    Parameters
    ----------
    tensor_desc : cudnnTensorDescriptor
        Previously allocated Tensor descriptor object.
    """

    assert _libcudnn
    status = _libcudnn.cudnnDestroyTensorDescriptor(tensor_desc)
    cudnnCheckStatus(status)


_libcudnn.cudnnTransformTensor.restype = int
_libcudnn.cudnnTransformTensor.argtypes = [
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_void_p,
]


def cudnnTransformTensor(
    handle: int,
    alpha: float,
    x_desc: int,
    x_data: ctypes.c_void_p,
    beta: float,
    y_desc: int,
    y_data: ctypes.c_void_p,
) -> None:
    """Tensor layout conversion helper (dest = alpha * src + beta * dest).

    This function copies the scaled data from one tensor to another tensor with a different
    layout. Those descriptors need to have the same dimensions but not necessarily the
    same strides. The input and output tensors must not overlap in any way (i.e., tensors
    cannot be transformed in place). This function can be used to convert a tensor with an
    unsupported tensor_format to a supported one.

    Parameters
    ----------
    handle : cudnnHandle
        cuDNN context.
    alpha : float
        Scalar factor to be applied to every element of the input tensor before it is added
        to the output tensor.
    x_desc : cudnnTensorDescriptor
        Handle to a previously initialized tensor descriptor.
    x_data : void_p
        Pointer to data of the tensor described by x_desc descriptor.
    beta: float
        Scaling factor which is applied on every element of the output tensor prior to adding
        the result of the operation. Note that if beta is zero, the output is not read and can
        contain any uninitialized data (including Nan numbers).
    y_desc : cudnnTensorDescriptor
        Handle to a previously initialized tensor descriptor.
    y_data : void_p
        Pointer to data of the tensor described by y_desc descriptor.
    """

    data_type, _, _, _, _, _, _, _, _ = cudnnGetTensor4dDescriptor(y_desc)
    if data_type == cudnnDataType["CUDNN_DATA_DOUBLE"]:
        alpha_ref = ctypes.byref(ctypes.c_double(alpha))
        beta_ref = ctypes.byref(ctypes.c_double(beta))
    else:
        alpha_ref = ctypes.byref(ctypes.c_float(alpha))
        beta_ref = ctypes.byref(ctypes.c_float(beta))

    assert _libcudnn
    status = _libcudnn.cudnnTransformTensor(
        handle, alpha_ref, x_desc, x_data, beta_ref, y_desc, y_data
    )
    cudnnCheckStatus(status)


_libcudnn.cudnnAddTensor.restype = int
_libcudnn.cudnnAddTensor.argtypes = [
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_void_p,
]


def cudnnAddTensor(
    handle: int,
    alpha: float,
    bias_desc: int,
    bias_data: ctypes.c_void_p,
    beta: float,
    src_dest_desc: int,
    src_dest_data: ctypes.c_void_p,
) -> None:
    """Tensor Bias addition : srcDest = alpha * bias + beta * src_dest_desc.

    This function adds the scaled values of one tensor to another tensor. The amount
    of data described by the bias_desc descriptor must match exactly the amount of data
    needed to perform the addition.

    Parameters
    ----------
    handle : cudnnHandle
        Handle to a cuDNN context.
    alpha : float
        Scalar factor to be applied to every data element of the bias tensor before it is added
        to the output tensor.
    bias_desc : cudnnTensorDescriptor
        Handle to a previously initialized tensor descriptor.
    bias_data : void_p
        Pointer to data of the tensor described by bias_desc.
    beta: float
        Scaling factor which is applied on every element of the output tensor prior to adding
        the result of the operation. Note that if beta is zero, the output is not read and can
        contain any uninitialized data (including Nan numbers).
    src_dest_desc : cudnnTensorDescriptor
        Handle to a previously initialized tensor descriptor.
    src_dest_data : void_p
        Pointer to data of the tensor described by src_dest_desc.
    """

    data_type, _, _, _, _, _, _, _, _ = cudnnGetTensor4dDescriptor(src_dest_desc)
    if data_type == cudnnDataType["CUDNN_DATA_DOUBLE"]:
        alpha_ref = ctypes.byref(ctypes.c_double(alpha))
        beta_ref = ctypes.byref(ctypes.c_double(beta))
    else:
        alpha_ref = ctypes.byref(ctypes.c_float(alpha))
        beta_ref = ctypes.byref(ctypes.c_float(beta))

    assert _libcudnn
    status = _libcudnn.cudnnAddTensor(
        handle, alpha_ref, bias_desc, bias_data, beta_ref, src_dest_desc, src_dest_data
    )
    cudnnCheckStatus(status)


_libcudnn.cudnnSetTensor.restype = int
_libcudnn.cudnnSetTensor.argtypes = [
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_void_p,
]


def cudnnSetTensor(handle: int, y_desc: int, y_data: ctypes.c_void_p, value: float) -> None:
    """Set all data points of a tensor to a given value : srcDest = value.

    Parameters
    ----------
    handle : cudnnHandle
        Handle to a previously created cuDNN context.
    y_desc : cudnnTensorDescriptor
        Handle to a previously initialized tensor descriptor.
    y_data : void_p
        Pointer to data of the tensor described by y_desc descriptor.
    value : float
        Value that all elements of the tensor will be set to.
    """

    data_type, _, _, _, _, _, _, _, _ = cudnnGetTensor4dDescriptor(y_desc)
    if data_type == cudnnDataType["CUDNN_DATA_DOUBLE"]:
        value_ref = ctypes.byref(ctypes.c_double(value))
    else:
        value_ref = ctypes.byref(ctypes.c_float(value))

    assert _libcudnn
    status = _libcudnn.cudnnSetTensor(handle, y_desc, y_data, value_ref)
    cudnnCheckStatus(status)


_libcudnn.cudnnScaleTensor.restype = int
_libcudnn.cudnnScaleTensor.argtypes = [
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_void_p,
]


def cudnnScaleTensor(handle: int, x_desc: int, x_data: ctypes.c_void_p, alpha: float) -> None:
    """This function scales all the elements of a tensor by a give factor.

    Set all data points of a tensor to scaled value : srcDest = alpha * srcDest.

    Parameters
    ----------
    handle : cudnnHandle
        Handle to a previously created cuDNN context.
    x_desc : cudnnTensorDescriptor
        Handle to a previously initialized tensor descriptor.
    x_data : void_p
        Pointer to data of the tensor described by x_desc descriptor.
    alpha : float
        Value that all elements of the tensor will be scaled with.
    """

    data_type, _, _, _, _, _, _, _, _ = cudnnGetTensor4dDescriptor(x_desc)
    if data_type == cudnnDataType["CUDNN_DATA_DOUBLE"]:
        alpha_ref = ctypes.byref(ctypes.c_double(alpha))
    else:
        alpha_ref = ctypes.byref(ctypes.c_float(alpha))

    assert _libcudnn
    status = _libcudnn.cudnnScaleTensor(handle, x_desc, x_data, alpha_ref)
    cudnnCheckStatus(status)


_libcudnn.cudnnCreateFilterDescriptor.restype = int
_libcudnn.cudnnCreateFilterDescriptor.argtypes = [ctypes.c_void_p]


def cudnnCreateFilterDescriptor() -> int:
    """Create a filter descriptor.

    This function creates a filter descriptor object by allocating the memory needed
    to hold its opaque structure.

    Parameters
    ----------
    Returns
    -------
    w_desc : cudnnFilterDescriptor
        Handle to a newly allocated filter descriptor.
    """  # noqa: D414

    w_desc = ctypes.c_void_p()
    assert _libcudnn
    status = _libcudnn.cudnnCreateFilterDescriptor(ctypes.byref(w_desc))
    cudnnCheckStatus(status)
    value = w_desc.value
    assert value
    return value


_libcudnn.cudnnSetFilter4dDescriptor.restype = int
_libcudnn.cudnnSetFilter4dDescriptor.argtypes = [
    ctypes.c_void_p,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
]


def cudnnSetFilter4dDescriptor(
    w_desc: int, data_type: int, tensor_format: int, k: int, c: int, h: int, w: int
) -> None:
    """Initialize a filter descriptor.

    This function initializes a previously created filter descriptor object into a 4D filter.
    Filters layout must be contiguous in memory.

    Parameters
    ----------
    w_desc : cudnnFilterDescriptor
        Handle to a previously created filter descriptor.
    data_type : cudnnDataType
        Data type.
    tensor_format: cudnnTensorFormat
        Tensor tensor_format
    k : int
        Number of output feature maps.
    c : int
        Number of input feature maps.
    h : int
        Height of each filter.
    w : int
        Width of each filter.
    """

    assert _libcudnn
    status = _libcudnn.cudnnSetFilter4dDescriptor(w_desc, data_type, tensor_format, k, c, h, w)
    cudnnCheckStatus(status)


_libcudnn.cudnnGetFilter4dDescriptor.restype = int
_libcudnn.cudnnGetFilter4dDescriptor.argtypes = [
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_void_p,
]


def cudnnGetFilter4dDescriptor(w_desc: int) -> tuple[int, int, int, int, int, int]:
    """Get parameters of filter descriptor.

    This function queries the parameters of the previously initialized filter descriptor object.

    Parameters
    ----------
    w_desc : cudnnFilterDescriptor
        Handle to a previously created filter descriptor.
    Returns
    -------
    data_type : cudnnDataType
        Data type.
    tensor_format: cudnnTensorFormat
        Tensor tensor_format
    k : int
        Number of output feature maps.
    c : int
        Number of input feature maps.
    h : int
        Height of each filter.
    w : int
        Width of each filter.
    """

    data_type = ctypes.c_int()
    tensor_format = ctypes.c_int()
    k = ctypes.c_int()
    c = ctypes.c_int()
    h = ctypes.c_int()
    w = ctypes.c_int()

    assert _libcudnn
    status = _libcudnn.cudnnGetFilter4dDescriptor(
        w_desc,
        ctypes.byref(data_type),
        ctypes.byref(tensor_format),
        ctypes.byref(k),
        ctypes.byref(c),
        ctypes.byref(h),
        ctypes.byref(w),
    )
    cudnnCheckStatus(status)

    return data_type.value, tensor_format.value, k.value, c.value, h.value, w.value


_libcudnn.cudnnDestroyFilterDescriptor.restype = int
_libcudnn.cudnnDestroyFilterDescriptor.argtypes = [ctypes.c_void_p]


def cudnnDestroyFilterDescriptor(w_desc: int) -> None:
    """Destroy filter descriptor.

    This function destroys a previously created Tensor4D descriptor object.

    Parameters
    ----------
    w_desc : cudnnFilterDescriptor
    """

    assert _libcudnn
    status = _libcudnn.cudnnDestroyFilterDescriptor(w_desc)
    cudnnCheckStatus(status)


_libcudnn.cudnnCreateConvolutionDescriptor.restype = int
_libcudnn.cudnnCreateConvolutionDescriptor.argtypes = [ctypes.c_void_p]


def cudnnCreateConvolutionDescriptor() -> int:
    """Create a convolution descriptor.

    This function creates a convolution descriptor object by allocating the memory needed to
    hold its opaque structure.

    Returns
    -------
    conv_desc : cudnnConvolutionDescriptor
        Handle to newly allocated convolution descriptor.
    """

    conv_desc = ctypes.c_void_p()

    assert _libcudnn
    status = _libcudnn.cudnnCreateConvolutionDescriptor(ctypes.byref(conv_desc))
    cudnnCheckStatus(status)
    value = conv_desc.value
    assert value
    return value


_libcudnn.cudnnSetConvolution2dDescriptor.restype = int
_libcudnn.cudnnSetConvolution2dDescriptor.argtypes = [
    ctypes.c_void_p,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
]


def cudnnSetConvolution2dDescriptor(
    conv_desc: int,
    pad_h: int,
    pad_w: int,
    u: int,
    v: int,
    dilation_h: int,
    dilation_w: int,
    mode: int,
    compute_type: int,
) -> None:
    """Initialize a convolution descriptor.

    This function initializes a previously created convolution descriptor object into a 2D
    correlation. This function assumes that the tensor and filter descriptors corresponds
    to the forward convolution path and checks if their settings are valid. That same
    convolution descriptor can be reused in the backward path provided it corresponds to
    the same layer.

    Parameters
    ----------
    conv_desc : cudnnConvolutionDescriptor
        Handle to a previously created convolution descriptor.
    pad_h : int
        zero-padding height: number of rows of zeros implicitly concatenated
        onto the top and onto the bottom of input images.
    pad_w : int
        zero-padding width: number of columns of zeros implicitly concatenated
        onto the left and onto the right of input images.
    u : int
        Vertical filter stride.
    v : int
        Horizontal filter stride.
    dilation_h : int
        Filter height dilation.
    dilation_w : int
        Filter width dilation.
    mode : cudnnConvolutionMode
        Select between CUDNN_CONVOLUTION or CUDNN_CROSS_CORRELATION.
    compute_type : cudnnDataType
        Compute precision
    """

    assert _libcudnn
    status = _libcudnn.cudnnSetConvolution2dDescriptor(
        conv_desc, pad_h, pad_w, u, v, dilation_h, dilation_w, mode, compute_type
    )
    cudnnCheckStatus(status)


_libcudnn.cudnnGetConvolution2dDescriptor.restype = int
_libcudnn.cudnnGetConvolution2dDescriptor.argtypes = [ctypes.c_void_p]


def cudnnGetConvolution2dDescriptor(
    conv_desc: int,
) -> tuple[int, int, int, int, int, int, int, int]:
    """Get a convolution descriptor.

    This function queries a previously initialized 2D convolution descriptor object.

    Parameters
    ----------
    conv_desc : cudnnConvolutionDescriptor
        Handle to a previously created convolution descriptor.
    Returns
    -------
    pad_h : int
        zero-padding height: number of rows of zeros implicitly concatenated onto
        the top and onto the bottom of input images.
    pad_w : int
        zero-padding width: number of columns of zeros implicitly concatenated
        onto the left and onto the right of input images.
    u : int
        Vertical filter stride.
    v : int
        Horizontal filter stride.
    dilation_h : int
        Filter height dilation.
    dilation_w : int
        Filter width dilation.
    mode : cudnnConvolutionMode
        Either CUDNN_CONVOLUTION or CUDNN_CROSS_CORRELATION.
    compute_type : cudnnDataType
        Compute precision
    """
    pad_h = ctypes.c_int()
    pad_w = ctypes.c_int()
    u = ctypes.c_int()
    v = ctypes.c_int()
    dilation_h = ctypes.c_int()
    dilation_w = ctypes.c_int()
    mode = ctypes.c_int()
    compute_type = ctypes.c_int()

    assert _libcudnn
    status = _libcudnn.cudnnGetConvolution2dDescriptor(
        conv_desc,
        ctypes.byref(pad_h),
        ctypes.byref(pad_w),
        ctypes.byref(u),
        ctypes.byref(v),
        ctypes.byref(dilation_h),
        ctypes.byref(dilation_w),
        ctypes.byref(mode),
        ctypes.byref(compute_type),
    )

    cudnnCheckStatus(status)

    return (
        pad_h.value,
        pad_w.value,
        u.value,
        v.value,
        dilation_h.value,
        dilation_w.value,
        mode.value,
        compute_type.value,
    )


_libcudnn.cudnnGetConvolution2dForwardOutputDim.restype = int
_libcudnn.cudnnGetConvolution2dForwardOutputDim.argtypes = [
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_void_p,
]


def cudnnGetConvolution2dForwardOutputDim(
    conv_desc: int, input_tensor_desc: int, w_desc: int
) -> tuple[int, int, int, int]:
    """Return the dimensions of the output tensor given a convolution descriptor.

    This function returns the dimensions of the resulting 4D tensor of a 2D
    convolution, given the convolution descriptor, the input tensor descriptor and
    the filter descriptor. This function can help to setup the output tensor and allocate
    the proper amount of memory prior to launching the actual convolution.

    Parameters
    ----------
    conv_desc : cudnnConvolutionDescriptor
        Handle to a previously created convolution descriptor.
    input_tensor_desc: cudnnTensorDescriptor
        Handle to a previously initialized tensor descriptor.
    w_desc: cudnnFilterDescriptor
        Handle to a previously initialized filter descriptor.
    Returns
    -------
    n : int
        Number of output images.
    c : int
        Number of output feature maps per image.
    h : int
        Height of each output feature map.
    w : int
        Width of each output feature map.
    """
    n = ctypes.c_int()
    c = ctypes.c_int()
    h = ctypes.c_int()
    w = ctypes.c_int()

    assert _libcudnn
    status = _libcudnn.cudnnGetConvolution2dForwardOutputDim(
        conv_desc,
        input_tensor_desc,
        w_desc,
        ctypes.byref(n),
        ctypes.byref(c),
        ctypes.byref(h),
        ctypes.byref(w),
    )
    cudnnCheckStatus(status)

    return n.value, c.value, h.value, w.value


_libcudnn.cudnnSetConvolutionNdDescriptor.restype = int
_libcudnn.cudnnSetConvolutionNdDescriptor.argtypes = [
    ctypes.c_void_p,  # conv_desc
    ctypes.c_int,  # arrayLength
    ctypes.POINTER(ctypes.c_int),  # pad_a[]
    ctypes.POINTER(ctypes.c_int),  # filter_stride_a[]
    ctypes.POINTER(ctypes.c_int),  # dilation_a[]
    ctypes.c_int,  # mode
    ctypes.c_int,
]  # data_type


def cudnnSetConvolutionNdDescriptor(
    conv_desc: int,
    pad_a: tuple[int, ...],
    filter_stride_a: tuple[int, ...],
    dilation_a: tuple[int, ...],
    mode: int,
    data_type: int,
) -> None:
    """Initialize a N-dimensional convolution descriptor.

    This function initializes a previously created convolution descriptor object into an N-D
    convolution. This function assumes that the tensor and filter descriptors corresponds
    to the forward convolution path and checks if their settings are valid. That same
    convolution descriptor can be reused in the backward path provided it corresponds to
    the same layer.

    Parameters
    ----------
    conv_desc : cudnnConvolutionDescriptor
        Handle to a previously created convolution descriptor.
    pad_a : int[]
        Array of padding values for each dimension.
    filter_stride_a : int[]
        Array of filter strides for each dimension.
    dilation_a : int[]
        Array of dilation values for each dimension.
    mode : cudnnConvolutionMode
        Select between CUDNN_CONVOLUTION or CUDNN_CROSS_CORRELATION.
    data_type : cudnnDataType
        Compute precision.
    """
    dim = len(pad_a)
    assert _libcudnn
    status = _libcudnn.cudnnSetConvolutionNdDescriptor(
        conv_desc,
        dim,
        (ctypes.c_int * dim)(*pad_a),
        (ctypes.c_int * dim)(*filter_stride_a),
        (ctypes.c_int * dim)(*dilation_a),
        mode,
        data_type,
    )
    cudnnCheckStatus(status)


_libcudnn.cudnnDestroyConvolutionDescriptor.restype = int
_libcudnn.cudnnDestroyConvolutionDescriptor.argtypes = [ctypes.c_void_p]


def cudnnDestroyConvolutionDescriptor(conv_desc: int) -> None:
    """Destroy a convolution descriptor.

    This function destroys a previously created convolution descriptor object.

    Parameters
    ----------
    conv_desc : int
        Previously created convolution descriptor.
    """

    assert _libcudnn
    status = _libcudnn.cudnnDestroyConvolutionDescriptor(conv_desc)
    cudnnCheckStatus(status)


class CudnnConvolutionFwdAlgoPerf(ctypes.Structure):
    """Performance result structure for forward convolution algorithms."""

    _fields_ = [
        ("algo", ctypes.c_int),
        ("status", ctypes.c_int),
        ("time", ctypes.c_float),
        ("memory", ctypes.c_size_t),
    ]

    def __str__(self) -> str:
        """Performance result structure for forward convolution algorithms representation."""
        return "(algo=%d, status=%d, time=%f, memory=%d)" % (
            self.algo,
            self.status,
            self.time,
            self.memory,
        )

    def __repr__(self) -> str:
        """Performance result structure for forward convolution algorithms representation."""
        return self.__str__()


_libcudnn.cudnnFindConvolutionForwardAlgorithm.restype = int
_libcudnn.cudnnFindConvolutionForwardAlgorithm.argtypes = [
    ctypes.c_void_p,  # handle
    ctypes.c_void_p,  # x_desc
    ctypes.c_void_p,  # w_desc
    ctypes.c_void_p,  # conv_desc
    ctypes.c_void_p,  # y_desc
    ctypes.c_int,  # requestAlgoCount
    ctypes.c_void_p,  # returned_algo_count
    ctypes.c_void_p,
]  # perf_results


def cudnnFindConvolutionForwardAlgorithm(
    handle: int, x_desc: int, w_desc: int, conv_desc: int, y_desc: int, requested_algo_count: int
) -> list[CudnnConvolutionFwdAlgoPerf]:
    """Find the best algorithm for forward convolution.

    This function searches for the best algorithm to execute the forward convolution operation
    given the input tensor descriptor, filter descriptor, convolution descriptor, and output tensor descriptor.
    It returns a list of performance results for the requested number of algorithms.

    Parameters
    ----------
    handle : cudnnHandle
        Handle to a previously created cuDNN context.
    x_desc : cudnnTensorDescriptor
        Handle to the previously initialized input tensor descriptor.
    w_desc : cudnnFilterDescriptor
        Handle to the previously initialized filter descriptor.
    conv_desc : cudnnConvolutionDescriptor
        Previously initialized convolution descriptor.
    y_desc : cudnnTensorDescriptor
        Handle to the previously initialized output tensor descriptor.
    requested_algo_count : int
        The number of algorithms to find.
    Returns
    -------
    perf_results : list of CudnnConvolutionFwdAlgoPerf
        A list of performance results for the found algorithms.
    """
    perf_results_type = CudnnConvolutionFwdAlgoPerf * requested_algo_count
    perf_results = perf_results_type()
    returned_algo_count = ctypes.c_int()
    assert _libcudnn
    status = _libcudnn.cudnnFindConvolutionForwardAlgorithm(
        handle,
        x_desc,
        w_desc,
        conv_desc,
        y_desc,
        ctypes.c_int(requested_algo_count),
        ctypes.byref(returned_algo_count),
        ctypes.cast(perf_results, ctypes.POINTER(CudnnConvolutionFwdAlgoPerf)),
    )
    cudnnCheckStatus(status)
    return perf_results[0: returned_algo_count.value]


# _libcudnn.cudnnGetConvolutionForwardAlgorithm.restype = int
# _libcudnn.cudnnGetConvolutionForwardAlgorithm.argtypes = [
#     ctypes.c_void_p,
#     ctypes.c_void_p,
#     ctypes.c_void_p,
#     ctypes.c_void_p,
#     ctypes.c_void_p,
#     ctypes.c_int,
#     ctypes.c_size_t,
#     ctypes.c_void_p
# ]
#
#
# def cudnnGetConvolutionForwardAlgorithm(
#     handle: int, x_desc: int, w_desc: int, conv_desc: int, y_desc: int, preference: int, memoryLimitInbytes: int
# ) -> int:
#     """Find the best algorithm for forward convolution.
#
#     This function returns the best algorithm to choose for the forward convolution
#     depending on the criteria expressed in the cudnnConvolutionFwdPreference_t enumerant.
#
#     Parameters
#     handle : cudnnHandle
#         Handle to a previously created cuDNN context.
#     x_desc : cudnnTensorDescriptor
#         Handle to a previously initialized tensor descriptor.
#     w_desc : cudnnFilterDescriptor
#         Handle to a previously initialized filter descriptor.
#     conv_desc : cudnnConvolutionDescriptor
#         Previously initialized convolution descriptor.
#     y_desc : cudnnTensorDescriptor
#         Handle to a previously initialized tensor descriptor.
#     preference : cudnnConvolutionFwdPreference
#         Enumerant to express the preference criteria in terms of memory
#         requirement and speed.
#     memoryLimitInbytes: size_t
#         The maximum amount of GPU memory the user is willing to use as a workspace
#         when preference is CUDNN_CONVOLUTION_FWD_SPECIFY_WORKSPACE_LIMIT.
#     Returns
#     algo: cudnnConvolutionFwdAlgo
#         Enumerant that specifies which convolution algorithm should be used to
#         compute the results according to the specified preference.
#     """
#     algo = ctypes.c_int()
#
#     status = _libcudnn.cudnnGetConvolutionForwardAlgorithm(
#         handle,
#         x_desc,
#         w_desc,
#         conv_desc,
#         y_desc,
#         preference,
#         ctypes.c_size_t(memoryLimitInbytes),
#         ctypes.byref(algo)
#     )
#     cudnnCheckStatus(status)
#
#     return algo.value


_libcudnn.cudnnSetConvolutionGroupCount.restype = int
_libcudnn.cudnnSetConvolutionGroupCount.argtypes = [ctypes.c_void_p, ctypes.c_int]


def cudnnSetConvolutionGroupCount(conv_desc: int, group_count: int) -> None:
    """This function allows the user to specify the number of groups to be used in the associated convolution.

    Parameters
    ----------
    conv_desc : cudnnConvolutionDescriptor
        Handle to a previously created convolution descriptor.
    group_count : int
        Number of groups to be used in the associated convolution.
    """
    assert _libcudnn
    status = _libcudnn.cudnnSetConvolutionGroupCount(conv_desc, group_count)

    cudnnCheckStatus(status)


_libcudnn.cudnnSetConvolutionMathType.restype = int
_libcudnn.cudnnSetConvolutionMathType.argtypes = [ctypes.c_void_p, ctypes.c_int]


def cudnnSetConvolutionMathType(conv_desc: int, math_type: int) -> None:
    """This function allows the user to specify whether or not the use of tensor op is permitted in the library routines associated with a given convolution descriptor.

    Parameters
    ----------
    conv_desc : cudnnConvolutionDescriptor
        Handle to a previously created convolution descriptor.
    math_type : int
        Number of groups to be used in the associated convolution.
    """
    assert _libcudnn
    status = _libcudnn.cudnnSetConvolutionMathType(conv_desc, math_type)

    cudnnCheckStatus(status)


_libcudnn.cudnnGetConvolutionForwardWorkspaceSize.restype = int
_libcudnn.cudnnGetConvolutionForwardWorkspaceSize.argtypes = [
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_int,
]


def cudnnGetConvolutionForwardWorkspaceSize(
    handle: int, x_desc: int, w_desc: int, conv_desc: int, y_desc: int, algo: int
) -> int:
    """This function returns the amount of GPU memory workspace the user needs to allocate to be able to call cudnnConvolutionForward with the specified algorithm.

    Parameters
    ----------
    handle : cudnnHandle
        Handle to a previously created cuDNN context.
    x_desc : cudnnTensorDescriptor
        Handle to a previously initialized tensor descriptor.
    w_desc : cudnnFilterDescriptor
        Handle to a previously initialized filter descriptor.
    conv_desc : cudnnConvolutionDescriptor
        Previously initialized convolution descriptor.
    y_desc : cudnnTensorDescriptor
        Handle to a previously initialized tensor descriptor.
    algo : cudnnConvolutionFwdAlgo
        Enumerant that specifies the chosen convolution algorithm.
    Returns
    -------
    size_in_bytes: c_size_t
        Amount of GPU memory needed as workspace to be able to execute a
        forward convolution with the specified algo.
    """
    size_in_bytes = ctypes.c_size_t()

    assert _libcudnn
    status = _libcudnn.cudnnGetConvolutionForwardWorkspaceSize(
        handle, x_desc, w_desc, conv_desc, y_desc, algo, ctypes.byref(size_in_bytes)
    )
    cudnnCheckStatus(status)

    return size_in_bytes.value


_libcudnn.cudnnConvolutionForward.restype = int
_libcudnn.cudnnConvolutionForward.argtypes = [
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_int,
    ctypes.c_void_p,
    ctypes.c_size_t,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_void_p,
]


def cudnnConvolutionForward(
    handle: int,
    alpha: float,
    x_desc: int,
    x_data: ctypes.c_void_p,
    w_desc: int,
    w: ctypes.c_void_p,
    conv_desc: int,
    algo: int,
    workspace: ctypes.c_void_p,
    workspace_size_in_bytes: int,
    beta: float,
    y_desc: int,
    y_data: ctypes.c_void_p,
) -> None:
    """Perform forward convolution. All of the form "output = alpha * Op(inputs) + beta * output".

    This function executes convolutions or cross-correlations over x using the specified
    filters, returning results in dest. Scaling factors alpha and beta can be used to scale
    the input tensor and the output tensor respectively.

    Parameters
    ----------
    handle : cudnnHandle
        Handle to a previously created cuDNN context.
    alpha: float
        Scaling factor with which every element of the input tensor is multiplied.
    x_desc : cudnnTensorDescriptor
        Handle to a previously initialized tensor descriptor.
    x_data : void_p
        Data pointer to GPU memory associated with the tensor descriptor x_desc.
    w_desc : cudnnFilterDescriptor
        Handle to a previously initialized filter descriptor.
    w : void_p
        Data pointer to GPU memory associated with the filter descriptor w_desc.
    conv_desc : cudnnConvolutionDescriptor
        Previously initialized convolution descriptor.
    algo: cudnnConvolutionFwdAlgo
        Enumerant that specifies which convolution algorithm should be used to
        compute the results.
    workspace: void_p
        Data pointer to GPU memory to a workspace needed to able to execute
        the specified algorithm. If no workspace is needed for a particular
        algorithm, that pointer can be nil.
    workspace_size_in_bytes: long
        Specifies the size in bytes of the provided workSpace.
    beta: float
        Scaling factor which is applied on every element of the output tensor prior
        to adding the result of the convolution.
    y_desc : cudnnTensorDescriptor
        Handle to a previously initialized tensor descriptor.
    y_data : void_p
        Data pointer to GPU memory associated with the tensor descriptor y_desc.
    """

    data_type = cudnnGetTensor4dDescriptor(y_desc)[0]
    if data_type == cudnnDataType["CUDNN_DATA_DOUBLE"]:
        alpha_ref = ctypes.byref(ctypes.c_double(alpha))
        beta_ref = ctypes.byref(ctypes.c_double(beta))
    else:
        alpha_ref = ctypes.byref(ctypes.c_float(alpha))
        beta_ref = ctypes.byref(ctypes.c_float(beta))

    assert _libcudnn
    status = _libcudnn.cudnnConvolutionForward(
        handle,
        alpha_ref,
        x_desc,
        x_data,
        w_desc,
        w,
        conv_desc,
        algo,
        workspace,
        ctypes.c_size_t(workspace_size_in_bytes),
        beta_ref,
        y_desc,
        y_data,
    )
    cudnnCheckStatus(status)


_libcudnn.cudnnConvolutionBackwardBias.restype = int
_libcudnn.cudnnConvolutionBackwardBias.argtypes = [
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_void_p,
]


def cudnnConvolutionBackwardBias(
    handle: int,
    alpha: float,
    dy_desc: int,
    dy_data: ctypes.c_void_p,
    beta: float,
    db_desc: int,
    db_data: ctypes.c_void_p,
) -> None:
    """Compute the gradient wrt the bias.

    This function computes the convolution gradient with respect to the bias, which is the
    sum of every element belonging to the same feature map across all of the images of the
    input tensor. Therefore, the number of elements produced is equal to the number of
    features maps of the input tensor.

    Parameters
    ----------
    handle : cudnnHandle
        Handle to a previously created cuDNN context.
    alpha: float
        Scaling factor with which every element of the input tensor is multiplied.
    dy_desc : cudnnTensorDescriptor
        Handle to the previously initialized input tensor descriptor.
    dy_data : void_p
        Data pointer to GPU memory associated with the tensor descriptor
        dy_desc.
    beta: float
        Scaling factor which is applied on every element of the output tensor prior
        to adding the result of the convolution gradient. Note that if beta is zero,
        the output is not read and can contain any uninitialized data (including
        Nan numbers).
    db_desc : cudnnTensorDescriptor
        Handle to the previously initialized output tensor descriptor.
    db_data : void_p
        Data pointer to GPU memory associated with the output tensor descriptor
        db_desc.
    """

    data_type = cudnnGetTensor4dDescriptor(db_desc)[0]
    if data_type == cudnnDataType["CUDNN_DATA_DOUBLE"]:
        alpha_ref = ctypes.byref(ctypes.c_double(alpha))
        beta_ref = ctypes.byref(ctypes.c_double(beta))
    else:
        alpha_ref = ctypes.byref(ctypes.c_float(alpha))
        beta_ref = ctypes.byref(ctypes.c_float(beta))

    assert _libcudnn
    status = _libcudnn.cudnnConvolutionBackwardBias(
        handle, alpha_ref, dy_desc, dy_data, beta_ref, db_desc, db_data
    )
    cudnnCheckStatus(status)


class CudnnConvolutionBwdDataAlgoPerf(ctypes.Structure):
    """Performance result structure for backward data convolution algorithms."""

    _fields_ = [
        ("algo", ctypes.c_int),
        ("status", ctypes.c_int),
        ("time", ctypes.c_float),
        ("memory", ctypes.c_size_t),
    ]

    def __str__(self) -> str:
        """Performance result structure for backward data convolution algorithms representation."""
        return "(algo=%d, status=%d, time=%f, memory=%d)" % (
            self.algo,
            self.status,
            self.time,
            self.memory,
        )

    def __repr__(self) -> str:
        """Performance result structure for backward data convolution algorithms representation."""
        return self.__str__()


_libcudnn.cudnnFindConvolutionBackwardDataAlgorithm.restype = int
_libcudnn.cudnnFindConvolutionBackwardDataAlgorithm.argtypes = [
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_int,
    ctypes.c_void_p,
    ctypes.c_void_p,
]


def cudnnFindConvolutionBackwardDataAlgorithm(
    handle: int, w_desc: int, dy_desc: int, conv_desc: int, dx_desc: int, requested_algo_count: int
) -> list[CudnnConvolutionBwdDataAlgoPerf]:
    """Find the best algorithm for backward data convolution.

    This function searches for the best algorithm to execute the backward data convolution operation
    given the filter descriptor, input differential tensor descriptor, convolution descriptor,
    and output differential tensor descriptor. It returns a list of performance results for the
    requested number of algorithms.

    Parameters
    ----------
    handle : cudnnHandle
        Handle to a previously created cuDNN context.
    w_desc : cudnnFilterDescriptor
        Handle to the previously initialized filter descriptor.
    dy_desc : cudnnTensorDescriptor
        Handle to the previously initialized input differential tensor descriptor.
    conv_desc : cudnnConvolutionDescriptor
        Previously initialized convolution descriptor.
    dx_desc : cudnnTensorDescriptor
        Handle to the previously initialized output differential tensor descriptor.
    requested_algo_count : int
        The number of algorithms to find.
    Returns
    -------
    perf_results : list of CudnnConvolutionBwdDataAlgoPerf
        A list of performance results for the found algorithms.
    """
    perf_results_type = CudnnConvolutionBwdDataAlgoPerf * requested_algo_count
    perf_results = perf_results_type()
    returned_algo_count = ctypes.c_int()
    assert _libcudnn
    status = _libcudnn.cudnnFindConvolutionBackwardDataAlgorithm(
        handle,
        w_desc,
        dy_desc,
        conv_desc,
        dx_desc,
        ctypes.c_int(requested_algo_count),
        ctypes.byref(returned_algo_count),
        ctypes.cast(perf_results, ctypes.POINTER(CudnnConvolutionBwdDataAlgoPerf)),
    )
    cudnnCheckStatus(status)
    return perf_results[0: returned_algo_count.value]


# _libcudnn.cudnnGetConvolutionBackwardDataAlgorithm.restype = int
# _libcudnn.cudnnGetConvolutionBackwardDataAlgorithm.argtypes = [ctypes.c_void_p,
#                                                                ctypes.c_void_p,
#                                                                ctypes.c_void_p,
#                                                                ctypes.c_void_p,
#                                                                ctypes.c_void_p,
#                                                                ctypes.c_int,
#                                                                ctypes.c_size_t,
#                                                                ctypes.c_void_p]
#
#
# def cudnnGetConvolutionBackwardDataAlgorithm(handle: int, w_desc: int, dy_desc: int, conv_desc: int,
#                                              dx_desc: int, preference: int, memoryLimitInbytes: int) -> int:
#     """This function serves as a heuristic for obtaining the best suited algorithm for cudnnConvolutionBackwardData for the given layer specifications.
#
#     Based on the input preference, this function will either return the fastest algorithm or the fastest
#     algorithm within a given memory limit. For an exhaustive search for the fastest
#     algorithm, please use cudnnFindConvolutionBackwardDataAlgorithm.
#
#     Parameters
#     handle : cudnnHandle
#         Handle to a previously created cuDNN context.
#     wDesc : cudnnFilterDescriptor
#         Handle to a previously initialized filter descriptor.
#     dyDesc : cuddnTensorDescriptor
#         Handle to the previously initialized input differential tensor descriptor.
#     convDesc : cudnnConvolutionDescriptor
#         Previously initialized convolution descriptor.
#     dxDesc : cuddnTensorDescriptor
#         Handle to the previously initialized output tensor descriptor
#     preference : cudnnConvolutionBwdPreference
#         Enumerant to express the preference criteria in terms of memory requirement and speed.
#     memoryLimitInbytes : size_t
#         It is to specify the maximum amount of GPU memory the user is willing to use
#         as a workspace. This is currently a placeholder and is not used.
#     Returns
#     algo : cudnnConvolutionBwdPreference
#         Enumerant that specifies which convolution algorithm should be used to
#         compute the results according to the specified preference.
#     """
#     algo = ctypes.c_int()
#     status = _libcudnn.cudnnGetConvolutionBackwardDataAlgorithm(handle,
#                                                                 w_desc,
#                                                                 dy_desc,
#                                                                 conv_desc,
#                                                                 dx_desc,
#                                                                 preference,
#                                                                 ctypes.c_size_t(memoryLimitInbytes),
#                                                                 ctypes.byref(algo))
#     cudnnCheckStatus(status)
#     return algo

_libcudnn.cudnnGetConvolutionBackwardDataWorkspaceSize.restype = int
_libcudnn.cudnnGetConvolutionBackwardDataWorkspaceSize.argtypes = [
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_int,
    ctypes.c_void_p,
]


def cudnnGetConvolutionBackwardDataWorkspaceSize(
    handle: int, w_desc: int, dy_desc: int, conv_desc: int, dx_desc: int, algo: int
) -> int:
    """Get the workspace size for backward data convolution.

    This function returns the amount of GPU memory workspace required to execute
    the backward data convolution operation with the specified algorithm.

    Parameters
    ----------
    handle : cudnnHandle
        Handle to a previously created cuDNN context.
    w_desc : cudnnFilterDescriptor
        Handle to the previously initialized filter descriptor.
    dy_desc : cudnnTensorDescriptor
        Handle to the previously initialized input differential tensor descriptor.
    conv_desc : cudnnConvolutionDescriptor
        Previously initialized convolution descriptor.
    dx_desc : cudnnTensorDescriptor
        Handle to the previously initialized output differential tensor descriptor.
    algo : cudnnConvolutionBwdDataAlgo
        Enumerant that specifies the chosen backward data convolution algorithm.
    Returns
    -------
    size_in_bytes : c_size_t
        Amount of GPU memory needed as workspace.
    """
    size_in_bytes = ctypes.c_size_t()
    assert _libcudnn
    status = _libcudnn.cudnnGetConvolutionBackwardDataWorkspaceSize(
        handle, w_desc, dy_desc, conv_desc, dx_desc, algo, ctypes.byref(size_in_bytes)
    )
    cudnnCheckStatus(status)
    return size_in_bytes.value


_libcudnn.cudnnConvolutionBackwardData.restype = int
_libcudnn.cudnnConvolutionBackwardData.argtypes = [
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_int,
    ctypes.c_void_p,
    ctypes.c_size_t,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_void_p,
]


def cudnnConvolutionBackwardData(
    handle: int,
    alpha: float,
    w_desc: int,
    w: ctypes.c_void_p,
    dy_desc: int,
    dy: ctypes.c_void_p,
    conv_desc: int,
    algo: int,
    workspace: ctypes.c_void_p,
    workspace_size_in_bytes: int,
    beta: float,
    dx_desc: int,
    dx: ctypes.c_void_p,
) -> None:
    """Perform backward data convolution.

    This function computes the gradient of the convolution operation with respect to the input data.
    The operation is of the form "output = alpha * Op(inputs) + beta * output".

    Parameters
    ----------
    handle : cudnnHandle
        Handle to a previously created cuDNN context.
    alpha : float
        Scaling factor with which every element of the input tensor is multiplied.
    w_desc : cudnnFilterDescriptor
        Handle to the previously initialized filter descriptor.
    w : void_p
        Data pointer to GPU memory associated with the filter descriptor w_desc.
    dy_desc : cudnnTensorDescriptor
        Handle to the previously initialized input differential tensor descriptor.
    dy : void_p
        Data pointer to GPU memory associated with the input differential tensor descriptor dy_desc.
    conv_desc : cudnnConvolutionDescriptor
        Previously initialized convolution descriptor.
    algo : cudnnConvolutionBwdDataAlgo
        Enumerant that specifies which backward data convolution algorithm should be used.
    workspace : void_p
        Data pointer to GPU memory for the workspace.
    workspace_size_in_bytes : size_t
        Specifies the size in bytes of the provided workSpace.
    beta : float
        Scaling factor which is applied on every element of the output tensor prior to adding
        the result of the convolution.
    dx_desc : cudnnTensorDescriptor
        Handle to the previously initialized output differential tensor descriptor.
    dx : void_p
        Data pointer to GPU memory associated with the output differential tensor descriptor dx_desc.
    """

    data_type = cudnnGetTensor4dDescriptor(dy_desc)[0]
    if data_type == cudnnDataType["CUDNN_DATA_FLOAT"]:
        alpha_ref = ctypes.byref(ctypes.c_float(alpha))
        beta_ref = ctypes.byref(ctypes.c_float(beta))
    else:
        alpha_ref = ctypes.byref(ctypes.c_double(alpha))
        beta_ref = ctypes.byref(ctypes.c_double(beta))

    assert _libcudnn
    status = _libcudnn.cudnnConvolutionBackwardData(
        handle,
        alpha_ref,
        w_desc,
        w,
        dy_desc,
        dy,
        conv_desc,
        algo,
        workspace,
        workspace_size_in_bytes,
        beta_ref,
        dx_desc,
        dx,
    )
    cudnnCheckStatus(status)


class CudnnConvolutionBwdFilterAlgoPerf(ctypes.Structure):
    """Performance result structure for backward filter convolution algorithms."""

    _fields_ = [
        ("algo", ctypes.c_int),
        ("status", ctypes.c_int),
        ("time", ctypes.c_float),
        ("memory", ctypes.c_size_t),
    ]

    def __str__(self) -> str:
        """Performance result structure for backward filter convolution algorithms representation."""
        return "(algo=%d, status=%d, time=%f, memory=%d)" % (
            self.algo,
            self.status,
            self.time,
            self.memory,
        )

    def __repr__(self) -> str:
        """Performance result structure for backward filter convolution algorithms representation."""
        return self.__str__()


_libcudnn.cudnnFindConvolutionBackwardFilterAlgorithm.restype = int
_libcudnn.cudnnFindConvolutionBackwardFilterAlgorithm.argtypes = [
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_int,
    ctypes.c_void_p,
    ctypes.c_void_p,
]


def cudnnFindConvolutionBackwardFilterAlgorithm(
    handle: int, x_desc: int, dy_desc: int, conv_desc: int, dw_desc: int, requested_algo_count: int
) -> list[CudnnConvolutionBwdFilterAlgoPerf]:
    """Find the best algorithm for backward filter convolution.

    This function searches for the best algorithm to execute the backward filter convolution operation
    given the input tensor descriptor, input differential tensor descriptor, convolution descriptor,
    and output differential filter descriptor. It returns a list of performance results for the requested
    number of algorithms.

    Parameters
    ----------
    handle : cudnnHandle
        Handle to a previously created cuDNN context.
    x_desc : cudnnTensorDescriptor
        Handle to the previously initialized input tensor descriptor.
    dy_desc : cudnnTensorDescriptor
        Handle to the previously initialized input differential tensor descriptor.
    conv_desc : cudnnConvolutionDescriptor
        Previously initialized convolution descriptor.
    dw_desc : cudnnTensorDescriptor
        Handle to the previously initialized output differential filter descriptor.
    requested_algo_count : int
        The number of algorithms to find.
    Returns
    -------
    perf_results : list of CudnnConvolutionBwdFilterAlgoPerf
        A list of performance results for the found algorithms.
    """
    perf_results_type = CudnnConvolutionBwdFilterAlgoPerf * requested_algo_count
    perf_results = perf_results_type()
    returned_algo_count = ctypes.c_int()
    assert _libcudnn
    status = _libcudnn.cudnnFindConvolutionBackwardFilterAlgorithm(
        handle,
        x_desc,
        dy_desc,
        conv_desc,
        dw_desc,
        ctypes.c_int(requested_algo_count),
        ctypes.byref(returned_algo_count),
        ctypes.cast(perf_results, ctypes.POINTER(CudnnConvolutionBwdFilterAlgoPerf)),
    )
    cudnnCheckStatus(status)
    return perf_results[0: returned_algo_count.value]


# _libcudnn.cudnnGetConvolutionBackwardFilterAlgorithm.restype = int
# _libcudnn.cudnnGetConvolutionBackwardFilterAlgorithm.argtypes = [ctypes.c_void_p,
#                                                                  ctypes.c_void_p,
#                                                                  ctypes.c_void_p,
#                                                                  ctypes.c_void_p,
#                                                                  ctypes.c_void_p,
#                                                                  ctypes.c_int,
#                                                                  ctypes.c_size_t,
#                                                                  ctypes.c_void_p]
#
#
# def cudnnGetConvolutionBackwardFilterAlgorithm(handle: int, x_desc: int, dy_desc: int, conv_desc: int,
#                                                dw_desc: int, preference: int, memoryLimitInbytes: int) -> int:
#     """This function serves as a heuristic for obtaining the best suited algorithm for cudnnConvolutionBackwardFilter for the given layer specifications.
#
#     Based on the input preference, this function will either return the fastest algorithm or the
#     fastest algorithm within a given memory limit. For an exhaustive search for the fastest
#     algorithm, please use cudnnFindConvolutionBackwardFilterAlgorithm.
#
#     Parameters
#     handle : cudnnHandle
#         Handle to a previously created cuDNN context.
#     x_desc : cuddnTensorDescriptor
#         Handle to the previously initialized input tensor descriptor.
#     dy_desc : cuddnTensorDescriptor
#         Handle to the previously initialized input differential tensor descriptor.
#     conv_desc : cudnnConvolutionDescriptor
#         Previously initialized convolution descriptor.
#     dw_desc : cudnnFilterDescriptor
#         Handle to a previously initialized filter descriptor.
#     preference : cudnnConvolutionBwdPreference
#         Enumerant to express the preference criteria in terms of memory requirement and speed.
#     memoryLimitInbytes : size_t
#         It is to specify the maximum amount of GPU memory the user is willing to use
#         as a workspace. This is currently a placeholder and is not used.
#     Returns
#     algo : cudnnConvolutionBwdPreference
#         Enumerant that specifies which convolution algorithm should be used to
#         compute the results according to the specified preference.
#     """
#     algo = ctypes.c_int()
#     status = _libcudnn.cudnnGetConvolutionBackwardFilterAlgorithm(handle,
#                                                                   x_desc,
#                                                                   dy_desc,
#                                                                   conv_desc,
#                                                                   dw_desc,
#                                                                   preference,
#                                                                   ctypes.c_size_t(memoryLimitInbytes),
#                                                                   ctypes.byref(algo))
#     cudnnCheckStatus(status)
#     return algo


_libcudnn.cudnnGetConvolutionBackwardFilterWorkspaceSize.restype = int
_libcudnn.cudnnGetConvolutionBackwardFilterWorkspaceSize.argtypes = [
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_int,
    ctypes.c_void_p,
]


def cudnnGetConvolutionBackwardFilterWorkspaceSize(
    handle: int, x_desc: int, dy_desc: int, conv_desc: int, grad_desc: int, algo: int
) -> int:
    """Get the workspace size for backward filter convolution.

    This function returns the amount of GPU memory workspace required to execute
    the backward filter convolution operation with the specified algorithm.

    Parameters
    ----------
    handle : cudnnHandle
        Handle to a previously created cuDNN context.
    x_desc : cudnnTensorDescriptor
        Handle to the previously initialized input tensor descriptor.
    dy_desc : cudnnTensorDescriptor
        Handle to the previously initialized input differential tensor descriptor.
    conv_desc : cudnnConvolutionDescriptor
        Previously initialized convolution descriptor.
    grad_desc : cudnnTensorDescriptor
        Handle to the previously initialized output differential filter descriptor.
    algo : cudnnConvolutionBwdFilterAlgo
        Enumerant that specifies the chosen backward filter convolution algorithm.
    Returns
    -------
    size_in_bytes : c_size_t
        Amount of GPU memory needed as workspace.
    """
    size_in_bytes = ctypes.c_size_t()
    assert _libcudnn
    status = _libcudnn.cudnnGetConvolutionBackwardFilterWorkspaceSize(
        handle, x_desc, dy_desc, conv_desc, grad_desc, algo, ctypes.byref(size_in_bytes)
    )
    cudnnCheckStatus(status)
    return size_in_bytes.value


_libcudnn.cudnnConvolutionBackwardFilter.restype = int
_libcudnn.cudnnConvolutionBackwardFilter.argtypes = [
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_int,
    ctypes.c_void_p,
    ctypes.c_size_t,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_void_p,
]


def cudnnConvolutionBackwardFilter(
    handle: int,
    alpha: float,
    x_desc: int,
    x: ctypes.c_void_p,
    dy_desc: int,
    dy: ctypes.c_void_p,
    conv_desc: int,
    algo: int,
    workspace: ctypes.c_void_p,
    workspace_size_in_bytes: int,
    beta: float,
    dw_desc: int,
    dw: ctypes.c_void_p,
) -> None:
    """Perform backward filter convolution.

    This function computes the gradient of the convolution operation with respect to the filter weights.
    The operation is of the form "output = alpha * Op(inputs) + beta * output".

    Parameters
    ----------
    handle : cudnnHandle
        Handle to a previously created cuDNN context.
    alpha : float
        Scaling factor with which every element of the input tensor is multiplied.
    x_desc : cudnnTensorDescriptor
        Handle to the previously initialized input tensor descriptor.
    x : void_p
        Data pointer to GPU memory associated with the input tensor descriptor x_desc.
    dy_desc : cudnnTensorDescriptor
        Handle to the previously initialized input differential tensor descriptor.
    dy : void_p
        Data pointer to GPU memory associated with the input differential tensor descriptor dy_desc.
    conv_desc : cudnnConvolutionDescriptor
        Previously initialized convolution descriptor.
    algo : cudnnConvolutionBwdFilterAlgo
        Enumerant that specifies which backward filter convolution algorithm should be used.
    workspace : void_p
        Data pointer to GPU memory for the workspace.
    workspace_size_in_bytes : size_t
        Specifies the size in bytes of the provided workSpace.
    beta : float
        Scaling factor which is applied on every element of the output tensor prior to adding
        the result of the convolution.
    dw_desc : cudnnTensorDescriptor
        Handle to the previously initialized output differential filter descriptor.
    dw : void_p
        Data pointer to GPU memory associated with the output differential filter descriptor dw_desc.
    """

    data_type = cudnnGetTensor4dDescriptor(dy_desc)[0]
    if data_type == cudnnDataType["CUDNN_DATA_DOUBLE"]:
        alpha_ref = ctypes.byref(ctypes.c_double(alpha))
        beta_ref = ctypes.byref(ctypes.c_double(beta))
    else:
        alpha_ref = ctypes.byref(ctypes.c_float(alpha))
        beta_ref = ctypes.byref(ctypes.c_float(beta))

    assert _libcudnn
    status = _libcudnn.cudnnConvolutionBackwardFilter(
        handle,
        alpha_ref,
        x_desc,
        x,
        dy_desc,
        dy,
        conv_desc,
        algo,
        workspace,
        workspace_size_in_bytes,
        beta_ref,
        dw_desc,
        dw,
    )
    cudnnCheckStatus(status)


_libcudnn.cudnnSoftmaxForward.restype = int
_libcudnn.cudnnSoftmaxForward.argtypes = [
    ctypes.c_void_p,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_void_p,
]


def cudnnSoftmaxForward(
    handle: int,
    algorithm: int,
    mode: int,
    alpha: float,
    x_desc: int,
    x_data: ctypes.c_void_p,
    beta: float,
    y_desc: int,
    y_data: ctypes.c_void_p,
) -> None:
    """This routing computes the softmax function

    Parameters
    ----------
    handle : cudnnHandle
        Handle to a previously created cuDNN context.
    algorithm : cudnnSoftmaxAlgorithm
        Enumerant to specify the softmax algorithm.
    mode : cudnnSoftmaxMode
        Enumerant to specify the softmax mode.
    alpha: float
        Scaling factor with which every element of the input tensors is multiplied.
    x_desc : cudnnTensorDescriptor
        Handle to the previously initialized input tensor descriptor.
    x_data : void_p
        Data pointer to GPU memory associated with the tensor descriptor
        x_desc.
    beta: float
        Scaling factor which is applied on every element of the output tensor prior
        to adding the result of the activation Note that if beta is zero, the output
        is not read and can contain any uninitialized data (including Nan numbers).
    y_desc : cudnnTensorDescriptor
        Handle to the previously initialized output tensor descriptor.
    y_data : void_p
        Data pointer to GPU memory associated with the output tensor descriptor
        y_desc.
    """

    data_type = cudnnGetTensor4dDescriptor(y_desc)[0]
    if data_type == cudnnDataType["CUDNN_DATA_DOUBLE"]:
        alpha_ref = ctypes.byref(ctypes.c_double(alpha))
        beta_ref = ctypes.byref(ctypes.c_double(beta))
    else:
        alpha_ref = ctypes.byref(ctypes.c_float(alpha))
        beta_ref = ctypes.byref(ctypes.c_float(beta))

    assert _libcudnn
    status = _libcudnn.cudnnSoftmaxForward(
        handle, algorithm, mode, alpha_ref, x_desc, x_data, beta_ref, y_desc, y_data
    )
    cudnnCheckStatus(status)


_libcudnn.cudnnSoftmaxBackward.restype = int
_libcudnn.cudnnSoftmaxBackward.argtypes = [
    ctypes.c_void_p,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_void_p,
]


def cudnnSoftmaxBackward(
    handle: int,
    algorithm: int,
    mode: int,
    alpha: float,
    y_desc: int,
    y_data: ctypes.c_void_p,
    dy_desc: int,
    dy_data: ctypes.c_void_p,
    beta: float,
    dx_desc: int,
    dx_data: ctypes.c_void_p,
) -> None:
    """This routine computes the gradient of the softmax function.

    Parameters
    ----------
    handle : cudnnHandle
        Handle to a previously created cuDNN context.
    algorithm : cudnnSoftmaxAlgorithm
        Enumerant to specify the softmax algorithm.
    mode : cudnnSoftmaxMode
        Enumerant to specify the softmax mode.
    alpha: float
        Scaling factor with which every element of the input tensors is multiplied.
    y_desc : cudnnTensorDescriptor
        Handle to the previously initialized input tensor descriptor.
    y_data : void_p
        Data pointer to GPU memory associated with the tensor descriptor
        y_desc.
    dy_esc : cudnnTensorDescriptor
        Handle to the previously initialized input differential tensor descriptor.
    dy_data : void_p
        Data pointer to GPU memory associated with the tensor descriptor
        dy_data.
    beta: float
        Scaling factor which is applied on every element of the output tensor prior
        to adding the result of the activation Note that if beta is zero, the output
        is not read and can contain any uninitialized data (including Nan numbers).
    dx_desc : cudnnTensorDescriptor
        Handle to the previously initialized output differential tensor descriptor.
    dx_data : void_p
        Data pointer to GPU memory associated with the output tensor descriptor
        dx_desc.
    """

    data_type = cudnnGetTensor4dDescriptor(dx_desc)[0]
    if data_type == cudnnDataType["CUDNN_DATA_DOUBLE"]:
        alpha_ref = ctypes.byref(ctypes.c_double(alpha))
        beta_ref = ctypes.byref(ctypes.c_double(beta))
    else:
        alpha_ref = ctypes.byref(ctypes.c_float(alpha))
        beta_ref = ctypes.byref(ctypes.c_float(beta))

    assert _libcudnn
    status = _libcudnn.cudnnSoftmaxBackward(
        handle,
        algorithm,
        mode,
        alpha_ref,
        y_desc,
        y_data,
        dy_desc,
        dy_data,
        beta_ref,
        dx_desc,
        dx_data,
    )
    cudnnCheckStatus(status)


_libcudnn.cudnnCreateDropoutDescriptor.restype = int
_libcudnn.cudnnCreateDropoutDescriptor.argtypes = [ctypes.c_void_p]


def cudnnCreateDropoutDescriptor() -> int:
    """Create dropout descriptor.

    This function creates a dropout descriptor object by allocating the memory needed to
    hold its opaque structure,

    Returns
    -------
    dropout_esc : cudnnDropoutDescriptor
        Newly allocated dropout descriptor.
    """

    dropout_esc = ctypes.c_void_p()
    assert _libcudnn
    status = _libcudnn.cudnnCreateDropoutDescriptor(ctypes.byref(dropout_esc))
    cudnnCheckStatus(status)
    value = dropout_esc.value
    assert value
    return value


_libcudnn.cudnnSetDropoutDescriptor.restype = int
_libcudnn.cudnnSetDropoutDescriptor.argtypes = [
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_float,
    ctypes.c_void_p,
    ctypes.c_size_t,
    ctypes.c_ulonglong,
]


def cudnnSetDropoutDescriptor(
    drop_desc: int,
    handle: int,
    dropout: float,
    states: ctypes.c_void_p,
    state_size_in_bytes: int,
    seed: int,
) -> None:
    """Set dropout descriptor parameters.

    Parameters
    ----------
    drop_desc : cudnnDropoutDescriptor
        Handle to a previously created dropout descriptor.
    handle : cudnnHandle
        Handle to a previously created cuDNN context.
    dropout : float
        The dropout ratio.
    states : void_p
        Pointer to the dropout states buffer.
    state_size_in_bytes : size_t
        Size of the dropout states buffer in bytes.
    seed : unsigned long long
        Seed for the random number generator.
    """
    assert _libcudnn
    status = _libcudnn.cudnnSetDropoutDescriptor(
        drop_desc, handle, dropout, states, state_size_in_bytes, seed
    )
    cudnnCheckStatus(status)


_libcudnn.cudnnDropoutGetReserveSpaceSize.restype = int
_libcudnn.cudnnDropoutGetReserveSpaceSize.argtypes = [ctypes.c_void_p]


def cudnnDropoutGetReserveSpaceSize(x_desc: int) -> int:
    """This function is used to query the amount of reserve needed to run dropout with the input dimensions given by x_desc

    Returns
    -------
    The size in bytes
    """

    size_in_bytes = ctypes.c_size_t()

    assert _libcudnn
    status = _libcudnn.cudnnDropoutGetReserveSpaceSize(x_desc, ctypes.byref(size_in_bytes))
    cudnnCheckStatus(status)

    return size_in_bytes.value


_libcudnn.cudnnDropoutGetStatesSize.restype = int
_libcudnn.cudnnDropoutGetStatesSize.argtypes = [ctypes.c_void_p]


def cudnnDropoutGetStatesSize(handle: int) -> int:
    """This function is used to query the amount of space required to store the states of the random number generators used by cudnnDropoutForward() function

    Returns
    -------
    The size in bytes
    """

    size_in_bytes = ctypes.c_size_t()

    assert _libcudnn
    status = _libcudnn.cudnnDropoutGetStatesSize(handle, ctypes.byref(size_in_bytes))
    cudnnCheckStatus(status)

    return size_in_bytes.value


_libcudnn.cudnnDropoutForward.restype = int
_libcudnn.cudnnDropoutForward.argtypes = [
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_size_t,
]


def cudnnDropoutForward(
    handle: int,
    dropout_esc: int,
    x_desc: int,
    x: ctypes.c_void_p,
    y_desc: int,
    y: ctypes.c_void_p,
    reserve_space: ctypes.c_void_p,
    reserve_space_size_in_bytes: int,
) -> None:
    """Perform dropout forward pass.

    Parameters
    ----------
    handle : cudnnHandle
        Handle to a previously created cuDNN context.
    dropout_esc : cudnnDropoutDescriptor
        Handle to a previously created dropout descriptor.
    x_desc : cudnnTensorDescriptor
        Handle to the previously initialized input tensor descriptor.
    x : void_p
        Data pointer to GPU memory associated with the tensor descriptor x_desc.
    y_desc : cudnnTensorDescriptor
        Handle to the previously initialized output tensor descriptor.
    y : void_p
        Data pointer to GPU memory associated with the tensor descriptor y_desc.
    reserve_space : void_p
        Data pointer to GPU memory for the reserve space.
    reserve_space_size_in_bytes : size_t
        Size of the reserve space in bytes.
    """
    assert _libcudnn
    status = _libcudnn.cudnnDropoutForward(
        handle,
        dropout_esc,
        x_desc,
        x,
        y_desc,
        y,
        reserve_space,
        ctypes.c_size_t(reserve_space_size_in_bytes),
    )
    cudnnCheckStatus(status)


_libcudnn.cudnnDropoutBackward.restype = int
_libcudnn.cudnnDropoutBackward.argtypes = [
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_void_p,
]


def cudnnDropoutBackward(
    handle: int,
    dropout_esc: int,
    dy_desc: int,
    dy: ctypes.c_void_p,
    dx_desc: int,
    dx: ctypes.c_void_p,
    reserve_space: ctypes.c_void_p,
    reserve_space_size_in_bytes: int,
) -> None:
    """Perform dropout backward pass.

    Parameters
    ----------
    handle : cudnnHandle
        Handle to a previously created cuDNN context.
    dropout_esc : cudnnDropoutDescriptor
        Handle to a previously created dropout descriptor.
    dy_desc : cudnnTensorDescriptor
        Handle to the previously initialized input differential tensor descriptor.
    dy : void_p
        Data pointer to GPU memory associated with the tensor descriptor dy_desc.
    dx_desc : cudnnTensorDescriptor
        Handle to the previously initialized output differential tensor descriptor.
    dx : void_p
        Data pointer to GPU memory associated with the tensor descriptor dx_desc.
    reserve_space : void_p
        Data pointer to GPU memory for the reserve space.
    reserve_space_size_in_bytes : size_t
        Size of the reserve space in bytes.
    """
    assert _libcudnn
    status = _libcudnn.cudnnDropoutBackward(
        handle, dropout_esc, dy_desc, dy, dx_desc, dx, reserve_space, reserve_space_size_in_bytes
    )

    cudnnCheckStatus(status)


_libcudnn.cudnnCreatePoolingDescriptor.restype = int
_libcudnn.cudnnCreatePoolingDescriptor.argtypes = [ctypes.c_void_p]


def cudnnCreatePoolingDescriptor() -> int:
    """Create pooling descriptor.

    This function creates a pooling descriptor object by allocating the memory needed to
    hold its opaque structure,

    Returns
    -------
    pooling_desc : cudnnPoolingDescriptor
        Newly allocated pooling descriptor.
    """

    pooling_desc = ctypes.c_void_p()
    assert _libcudnn
    status = _libcudnn.cudnnCreatePoolingDescriptor(ctypes.byref(pooling_desc))
    cudnnCheckStatus(status)
    value = pooling_desc.value
    assert value
    return value


_libcudnn.cudnnSetPooling2dDescriptor.restype = int
_libcudnn.cudnnSetPooling2dDescriptor.argtypes = [
    ctypes.c_void_p,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
]


def cudnnSetPooling2dDescriptor(
    pooling_desc: int,
    mode: int,
    nan: int,
    window_height: int,
    window_width: int,
    vertical_padding: int,
    horizontal_padding: int,
    vertical_stride: int,
    horizontal_stride: int,
) -> None:
    """Initialize a 2D pooling descriptor.

    This function initializes a previously created pooling descriptor object.

    Parameters
    ----------
    pooling_desc : cudnnPoolingDescriptor
        Handle to a previously created pooling descriptor.
    nan: cudnnNanPropagation
        Enumerate to specify the nan propagation
    mode : cudnnPoolingMode
        Enumerant to specify the pooling mode.
    window_height : int
        Height of the pooling window.
    window_width : int
        Width of the pooling window.
    vertical_padding: int
        Size of vertical padding.
    horizontal_padding: int
        Size of horizontal padding.
    vertical_stride : int
        Pooling vertical stride.
    horizontal_stride : int
        Pooling horizontal stride.
    """

    assert _libcudnn
    status = _libcudnn.cudnnSetPooling2dDescriptor(
        pooling_desc,
        mode,
        nan,
        window_height,
        window_width,
        vertical_padding,
        horizontal_padding,
        vertical_stride,
        horizontal_stride,
    )
    cudnnCheckStatus(status)


_libcudnn.cudnnGetPooling2dDescriptor.restype = int
_libcudnn.cudnnGetPooling2dDescriptor.argtypes = [
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_void_p,
]


def cudnnGetPooling2dDescriptor(pooling_desc: int) -> tuple[int, int, int, int, int, int, int, int]:
    """This function queries a previously created pooling descriptor object.

    Parameters
    ----------
    pooling_desc : cudnnPoolingDescriptor
    Handle to a previously created 2D pooling descriptor.
    Returns
    -------
    mode : cudnnPoolingMode
        Enumerant to specify the pooling mode.
    maxpoolingNanOpt:
        Enumerant to specify the Nan propagation mode.
    window_height : int
        Height of the pooling window.
    window_width : int
        Width of the pooling window.
    vertical_padding: int
        Size of vertical padding.
    horizontal_padding: int
        Size of horizontal padding.
    vertical_stride : int
        Pooling vertical stride.
    horizontal_stride : int
        Pooling horizontal stride.
    https://docs.nvidia.com/deeplearning/cudnn/backend/latest/api/cudnn-ops-library.html#cudnngetpooling2ddescriptor
    """

    mode = ctypes.c_int()
    nan = ctypes.c_int()
    window_height = ctypes.c_int()
    window_width = ctypes.c_int()
    vertical_padding = ctypes.c_int()
    horizontal_padding = ctypes.c_int()
    vertical_stride = ctypes.c_int()
    horizontal_stride = ctypes.c_int()

    assert _libcudnn
    status = _libcudnn.cudnnGetPooling2dDescriptor(
        pooling_desc,
        ctypes.byref(mode),
        ctypes.byref(nan),
        ctypes.byref(window_height),
        ctypes.byref(window_width),
        ctypes.byref(vertical_padding),
        ctypes.byref(horizontal_padding),
        ctypes.byref(vertical_stride),
        ctypes.byref(horizontal_stride),
    )
    cudnnCheckStatus(status)
    return (
        mode.value,
        nan.value,
        window_height.value,
        window_width.value,
        vertical_padding.value,
        horizontal_padding.value,
        vertical_stride.value,
        horizontal_stride.value,
    )


_libcudnn.cudnnDestroyPoolingDescriptor.restype = int
_libcudnn.cudnnDestroyPoolingDescriptor.argtypes = [ctypes.c_void_p]


def cudnnDestroyPoolingDescriptor(pooling_desc: int) -> None:
    """This function destroys a previously created pooling descriptor object.

    Parameters
    ----------
    pooling_desc : cudnnPoolingDescriptor
    """

    assert _libcudnn
    status = _libcudnn.cudnnDestroyPoolingDescriptor(pooling_desc)
    cudnnCheckStatus(status)


_libcudnn.cudnnGetPooling2dForwardOutputDim.restype = int
_libcudnn.cudnnGetPooling2dForwardOutputDim.argtypes = [
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_void_p,
]


def cudnnGetPooling2dForwardOutputDim(
    pooling_desc: int, input_desc: int
) -> tuple[int, int, int, int]:
    """This function provides the output dimensions of a tensor after 2d pooling has been applied.

    Each dimension h and w of the output images is computed as follows:
        outputDim = 1 + (inputDim + 2*padding - windowDim)/poolingStride;

    Parameters
    ----------
    pooling_desc : cudnnPoolingDescriptor
        Handle to a previously initialized pooling descriptor.
    input_desc : cudnnTensorDescriptor
        Handle to the previously initialized input tensor descriptor.

    Returns
    -------
    n : int
        Number of images in the output.
    c : int
        Number of channels in the output.
    h : int
        Height of images in the output.
    w : int
        Width of images in the output.
    """
    n = ctypes.c_int()
    c = ctypes.c_int()
    h = ctypes.c_int()
    w = ctypes.c_int()

    assert _libcudnn
    status = _libcudnn.cudnnGetPooling2dForwardOutputDim(
        pooling_desc, input_desc, ctypes.byref(n), ctypes.byref(c), ctypes.byref(h), ctypes.byref(w)
    )
    cudnnCheckStatus(status)

    return n.value, c.value, h.value, w.value


_libcudnn.cudnnPoolingForward.restype = int
_libcudnn.cudnnPoolingForward.argtypes = [
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_void_p,
]


def cudnnPoolingForward(
    handle: int,
    pooling_desc: int,
    alpha: float,
    x_desc: int,
    x_data: ctypes.c_void_p,
    beta: float,
    y_desc: int,
    y_data: ctypes.c_void_p,
) -> None:
    """Perform pooling.

    This function computes pooling of input values (i.e., the maximum or average of several
    adjacent values) to produce an output with smaller height and/or width.

    Parameters
    ----------
    handle : cudnnHandle
        Handle to a previously created cuDNN context.
    pooling_desc : cudnnPoolingDescriptor
        Handle to a previously initialized pooling descriptor.
    alpha: float
        Scaling factor with which every element of the input tensor is multiplied.
    x_desc : cudnnTensorDescriptor
        Handle to the previously initialized input tensor descriptor.
    x_data : void_p
        Data pointer to GPU memory associated with the tensor descriptor
        x_desc.
    beta: float
        Scaling factor which is applied on every element of the output tensor prior
        to adding the result of the activation Note that if beta is zero, the output
        is not read and can contain any uninitialized data (including Nan numbers).
    y_desc : cudnnTensorDescriptor
        Handle to the previously initialized output tensor descriptor.
    y_data : void_p
        Data pointer to GPU memory associated with the output tensor descriptor
        y_desc.
    """

    data_type = cudnnGetTensor4dDescriptor(y_desc)[0]
    if data_type == cudnnDataType["CUDNN_DATA_DOUBLE"]:
        alpha_ref = ctypes.byref(ctypes.c_double(alpha))
        beta_ref = ctypes.byref(ctypes.c_double(beta))
    else:
        alpha_ref = ctypes.byref(ctypes.c_float(alpha))
        beta_ref = ctypes.byref(ctypes.c_float(beta))

    assert _libcudnn
    status = _libcudnn.cudnnPoolingForward(
        handle, pooling_desc, alpha_ref, x_desc, x_data, beta_ref, y_desc, y_data
    )
    cudnnCheckStatus(status)


_libcudnn.cudnnPoolingBackward.restype = int
_libcudnn.cudnnPoolingBackward.argtypes = [
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_void_p,
]


def cudnnPoolingBackward(
    handle: int,
    pooling_desc: int,
    alpha: float,
    y_desc: int,
    y_data: ctypes.c_void_p,
    dy_desc: int,
    dy_data: ctypes.c_void_p,
    x_desc: int,
    x_data: ctypes.c_void_p,
    beta: float,
    dx_desc: int,
    dx_data: ctypes.c_void_p,
) -> None:
    """Gradients wrt the pooling operation.

    This function computes the gradient of a pooling operation.
    Parameters
    ----------
    handle : cudnnHandle
        Handle to a previously created cuDNN context.
    pooling_desc : cudnnPoolingDescriptor
        Handle to the previously initialized pooling descriptor.
    alpha: float
        Scaling factor with which every element of the input tensors is multiplied.
    y_desc : cudnnTensorDescriptor
        Handle to the previously initialized input tensor descriptor.
    y_data : void_p
        Data pointer to GPU memory associated with the tensor descriptor
        y_desc.
    dy_esc : cudnnTensorDescriptor
        Handle to the previously initialized input differential tensor descriptor.
    dy_data : void_p
        Data pointer to GPU memory associated with the tensor descriptor
        dy_data.
    x_desc : cudnnTensorDescriptor
        Handle to the previously initialized output tensor descriptor.
    x_data : void_p
        Data pointer to GPU memory associated with the output tensor descriptor
        x_desc.
    beta: float
        Scaling factor which is applied on every element of the output tensor prior
        to adding the result of the activation Note that if beta is zero, the output
        is not read and can contain any uninitialized data (including Nan numbers).
    dx_desc : cudnnTensorDescriptor
        Handle to the previously initialized output differential tensor descriptor.
    dx_data : void_p
        Data pointer to GPU memory associated with the output tensor descriptor
        dx_desc.
    """

    data_type = cudnnGetTensor4dDescriptor(x_desc)[0]
    if data_type == cudnnDataType["CUDNN_DATA_DOUBLE"]:
        alpha_ref = ctypes.byref(ctypes.c_double(alpha))
        beta_ref = ctypes.byref(ctypes.c_double(beta))
    else:
        alpha_ref = ctypes.byref(ctypes.c_float(alpha))
        beta_ref = ctypes.byref(ctypes.c_float(beta))

    assert _libcudnn
    status = _libcudnn.cudnnPoolingBackward(
        handle,
        pooling_desc,
        alpha_ref,
        y_desc,
        y_data,
        dy_desc,
        dy_data,
        x_desc,
        x_data,
        beta_ref,
        dx_desc,
        dx_data,
    )
    cudnnCheckStatus(status)


_libcudnn.cudnnDeriveBNTensorDescriptor.restype = int
_libcudnn.cudnnDeriveBNTensorDescriptor.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int]


def cudnnDeriveBNTensorDescriptor(derive_bn_desc: int, x_desc: int, mode: int) -> None:
    """This function derives a secondary tensor descriptor for the batch normalization scale, invVariance, bn_bias, and bn_scale subtensors from the layer's x data descriptor.

    Parameters
    ----------
    derive_bn_desc : cudnnTensorDescriptor
        Handle to a previously created tensor descriptor.
    x_desc : cudnnTensorDescriptor
        Handle to a previously created and initialized layer's x data descriptor.
    mode : int
        Batch normalization layer mode of operation.
    """
    assert _libcudnn
    status = _libcudnn.cudnnDeriveBNTensorDescriptor(derive_bn_desc, x_desc, mode)

    cudnnCheckStatus(status)


_libcudnn.cudnnCreateSeqDataDescriptor.restype = int
_libcudnn.cudnnCreateSeqDataDescriptor.argtypes = [ctypes.c_void_p]


def cudnnCreateSeqDataDescriptor() -> int:
    """Create a SeqData descriptor object.

    Allocates a cudnnSeqDataDescriptor_t structure and returns a pointer to it.

    Returns
    -------
    seqdata_descriptor : int
        SeqData descriptor.
    """

    seq_data = ctypes.c_void_p()
    assert _libcudnn
    status = _libcudnn.cudnnCreateSeqDataDescriptor(ctypes.byref(seq_data))
    cudnnCheckStatus(status)
    value = seq_data.value
    assert value
    return value


_libcudnn.cudnnSetSeqDataDescriptor.restype = int
_libcudnn.cudnnSetSeqDataDescriptor.argtypes = [
    ctypes.c_void_p,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_size_t,
    ctypes.c_void_p,
    ctypes.c_void_p,
]


def cudnnSetSeqDataDescriptor(
    seq_data_desc: int,
    data_type: int,
    nb_dims: int,
    dim_a: tuple[int, ...],
    axes: tuple[int, ...],
    seq_length_array_size: int,
    seq_length_array: tuple[int, ...],
    padding_fill: None,
) -> None:
    """Initialize a previously created SeqData object.

    This function initializes a previously created sequence data descriptor object. In the most
    simplified view, this descriptor defines dimensions (dimA) and the data layout (axes) of a
    four-dimensional tensor. All four dimensions of the sequence data descriptor have unique
    identifiers that can be used to index the dimA[] array.

    Parameters
    ----------
    seq_data_desc : cudnnSeqDataDescriptor
        Pointer to a previously created sequence data descriptor.
    data_type : cudnnDataType
        Data type of the sequence data buffer (CUDNN_DATA_HALF, CUDNN_DATA_FLOAT or
        CUDNN_DATA_DOUBLE).
    nb_dims : int
        Must be 4. The number of active dimensions in dimA[] and axes[] arrays. Both arrays should
        be declared to contain at least CUDNN_SEQDATA_DIM_COUNT elements.
    dim_a : int[]
        Integer array specifying sequence data dimensions. Use the cudnnSeqDataAxis enumerated
        type to index all active dimA[] elements.
    axes : cudnnSeqDataAxis[]
        Array of cudnnSeqDataAxis that defines the layout of sequence data in memory. The first
        nbDims elements of axes[] should be initialized with the outermost dimension in axes[0] and
        the innermost dimension in axes[nbDims-1].
    seq_length_array_size : int
        Number of elements in the sequence length array, seqLengthArray[].
    seq_length_array : int[]
        An integer array that defines all sequence lengths of the container.
    padding_fill : void
        Must be NULL. Pointer to a value of dataType that is used to fill up output vectors beyond
        the valid length of each sequence or NULL to ignore this setting.
    """
    dim_a_ref = (ctypes.c_int32 * len(dim_a))(dim_a)
    axes_ref = (ctypes.c_int32 * len(axes))(axes)
    seq_length_array_ref = (ctypes.c_int32 * seq_length_array_size)(seq_length_array)
    assert _libcudnn
    status = _libcudnn.cudnnSetSeqDataDescriptor(
        seq_data_desc,
        data_type,
        nb_dims,
        dim_a_ref,
        axes_ref,
        seq_length_array_size,
        seq_length_array_ref,
        padding_fill,
    )
    cudnnCheckStatus(status)


_libcudnn.cudnnDestroySeqDataDescriptor.restype = int
_libcudnn.cudnnDestroySeqDataDescriptor.argtypes = [ctypes.c_void_p]


def cudnnDestroySeqDataDescriptor(seq_data_desc: int) -> None:
    """Destroy a SeqData descriptor.

    This function destroys a previously created SeqData descriptor object.

    Parameters
    ----------
    seq_data_desc : cudnnSeqDataDescriptor
        Previously allocated SeqData descriptor object.
    """

    assert _libcudnn
    status = _libcudnn.cudnnDestroySeqDataDescriptor(seq_data_desc)
    cudnnCheckStatus(status)


_libcudnn.cudnnCreateAttnDescriptor.restype = int
_libcudnn.cudnnCreateAttnDescriptor.argtypes = [ctypes.c_void_p]


def cudnnCreateAttnDescriptor() -> int:
    """Create a attnDesc descriptor object.

    Allocates a cudnnAttnDescriptor_t structure and returns a pointer to it.

    Returns
    -------
    cudnnAttnDescriptor : int
        attnDesc descriptor.
    """

    attn_desc = ctypes.c_void_p()
    assert _libcudnn
    status = _libcudnn.cudnnCreateAttnDescriptor(ctypes.byref(attn_desc))
    cudnnCheckStatus(status)
    value = attn_desc.value
    assert value
    return value


_libcudnn.cudnnSetAttnDescriptor.restype = int
_libcudnn.cudnnSetAttnDescriptor.argtypes = [
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_int,
    ctypes.c_double,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
]


def cudnnSetAttnDescriptor(
    attn_desc: int,
    attn_mode: int,
    n_heads: int,
    sm_scaler: float,
    data_type: int,
    compute_prec: int,
    math_type: int,
    attn_dropout_desc: int,
    post_dropout_desc: int,
    q_size: int,
    k_size: int,
    v_size: int,
    q_proj_size: int,
    k_proj_size: int,
    v_proj_size: int,
    o_proj_size: int,
    qo_Max_seq_length: int,
    kv_max_seq_length: int,
    max_batch_size: int,
    max_beam_size: int,
) -> None:
    """This function configures a multi-head attention descriptor that was previously created using the cudnnCreateAttnDescriptor() function.

    The function sets attention parameters that are
    necessary to compute internal buffer sizes, dimensions of weight and bias tensors, or to
    select optimized code paths.

    Parameters
    ----------
    attn_desc : cudnnAttnDescriptor
        Attention descriptor to be configured.
    attn_mode : unsigned
        Enables various attention options that do not require additional numerical values.
        The user should assign a preferred set of bitwise OR-ed flags to this argument.
    n_heads : int
        Number of attention heads.
    sm_scaler : double
        Softmax smoothing (1.0 >= smScaler >= 0.0) or sharpening (smScaler > 1.0) coefficient.
        Negative values are not accepted.
    data_type : cudnnDataType
        Data type used to represent attention inputs, attention weights and attention outputs.
    compute_prec : cudnnDataType
        Compute precision.
    math_type : cudnnMathType
        NVIDIA Tensor Core settings.
    attn_dropout_desc : cudnnDropoutDescriptor
        Descriptor of the dropout operation applied to the softmax output. See the table below
        for a list of unsupported features.
    post_dropout_desc : cudnnDropoutDescriptor
        Descriptor of the dropout operation applied to the multi-head attention output, just
        before the point where residual connections are added.
    q_size, k_size, v_size : int
        Q , K , V embedding vector lengths.
    q_proj_size, k_proj_size, v_proj_size : int
        Q , K , V embedding vector lengths after input projections. Use zero to disable the
        corresponding projection.
    o_proj_size : int
        The h i vector length after the output projection. Use zero to disable this projection.
    qo_max_seq_length : int
        Largest sequence length expected in sequence data descriptors related to Q , O , dQ
        and dO inputs and outputs.
    kv_max_seq_length : int
        Largest sequence length expected in sequence data descriptors related to K , V , dK
        and dV inputs and outputs.
    max_batch_size : int
        Largest batch size expected in any cudnnSeqDataDescriptor_t container.
    max_beam_size : int
        Largest beam size expected in any cudnnSeqDataDescriptor_t container.
    """
    assert _libcudnn
    status = _libcudnn.cudnnSetAttnDescriptor(
        attn_desc,
        attn_mode,
        n_heads,
        sm_scaler,
        data_type,
        compute_prec,
        math_type,
        attn_dropout_desc,
        post_dropout_desc,
        q_size,
        k_size,
        v_size,
        q_proj_size,
        k_proj_size,
        v_proj_size,
        o_proj_size,
        qo_Max_seq_length,
        kv_max_seq_length,
        max_batch_size,
        max_beam_size,
    )
    cudnnCheckStatus(status)


_libcudnn.cudnnDestroyAttnDescriptor.restype = int
_libcudnn.cudnnDestroyAttnDescriptor.argtypes = [ctypes.c_void_p]


def cudnnDestroyAttnDescriptor(attn_desc: int) -> None:
    """Destroy a Attn descriptor.

    This function destroys a previously created Attn descriptor object.

    Parameters
    ----------
    attn_desc : cudnnAttnDescriptor
        Previously allocated Attn descriptor object.
    """

    assert _libcudnn
    status = _libcudnn.cudnnDestroyAttnDescriptor(attn_desc)
    cudnnCheckStatus(status)


_libcudnn.cudnnGetMultiHeadAttnWeights.restype = int
_libcudnn.cudnnGetMultiHeadAttnWeights.argtypes = [
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_size_t,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_void_p,
]


def cudnnGetMultiHeadAttnWeights(
    handle: int, attn_desc: int, w_kind: int, weight_size_in_bytes: int, weights: ctypes.c_void_p
) -> tuple[int, ctypes.c_void_p]:
    """This function obtains the shape of the weight or bias tensor.

    It also retrieves the start address of tensor data located in the weight buffer.
    Use the wKind argument to select a particular tensor. For more information,
    see cudnnMultiHeadAttnWeightKind for the description of the enumerant type.

    Parameters
    ----------
    handle : cudnnHandle
        The current cuDNN context handle.
    attn_desc : cudnnAttnDescriptor
        A previously configured attention descriptor.
    w_kind : cudnnMultiHeadAttnWeightKind
        Enumerant type to specify which weight or bias tensor should be retrieved.
    weight_size_in_bytes : size_t
        Buffer size that stores all multi-head attention weights and biases.
    weights : void
    Input. Pointer to the weight buffer in the host or device memory.
    Returns
    -------
    w_desc : cudnnTensorDescriptor
        The descriptor specifying weight or bias tensor shape. For weights, the wDesc.dimA[] array has
        three elements: [nHeads, projected size, original size]. For biases, the wDesc.dimA[] array also
        has three elements: [nHeads, projected size, 1]. The wDesc.strideA[] array describes how tensor
        elements are arranged in memory.
    w_addr : void_p
        Pointer to a location where the start address of the requested tensor should be written. When the
        corresponding projection is disabled, the address written to wAddr is NULL.
    """
    w_desc = ctypes.c_void_p()
    assert _libcudnn
    status = _libcudnn.cudnnCreateTensorDescriptor(ctypes.byref(w_desc))
    w_addr = (ctypes.POINTER(ctypes.c_void_p) * 1)()

    assert _libcudnn
    status = _libcudnn.cudnnGetMultiHeadAttnWeights(
        handle, attn_desc, w_kind, weight_size_in_bytes, weights, w_desc, ctypes.byref(w_addr)
    )
    cudnnCheckStatus(status)
    w_addr = w_addr[0]
    value = w_desc.value
    assert value
    return value, w_addr


_libcudnn.cudnnGetMultiHeadAttnBuffers.restype = int
_libcudnn.cudnnGetMultiHeadAttnBuffers.argtypes = [
    ctypes.c_void_p,
    ctypes.c_void_p,
]


def cudnnGetMultiHeadAttnBuffers(handle: int, attn_desc: int) -> tuple[int, int, int]:
    """This function computes weight, work, and reserve space buffer sizes used by the following functions: cudnnMultiHeadAttnForward(), cudnnMultiHeadAttnBackwardData(), cudnnMultiHeadAttnBackwardWeights()

    Returns
    -------
    weightSizeInBytes : size_t
        The size in bytes
    workSpaceSizeInBytes : size_t
        The size in bytes
    reserveSpaceSizeInBytes : size_t
        The size in bytes
    """

    weight_size_in_bytes = ctypes.c_size_t()
    work_space_size_in_bytes = ctypes.c_size_t()
    reserve_space_size_in_bytes = ctypes.c_size_t()

    assert _libcudnn
    status = _libcudnn.cudnnGetMultiHeadAttnBuffers(
        handle,
        attn_desc,
        ctypes.byref(weight_size_in_bytes),
        ctypes.byref(work_space_size_in_bytes),
        ctypes.byref(reserve_space_size_in_bytes),
    )
    cudnnCheckStatus(status)

    return (
        weight_size_in_bytes.value,
        work_space_size_in_bytes.value,
        reserve_space_size_in_bytes.value,
    )


_libcudnn.cudnnMultiHeadAttnForward.restype = int
_libcudnn.cudnnMultiHeadAttnForward.argtypes = [
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_int,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_size_t,
    ctypes.c_void_p,
    ctypes.c_size_t,
    ctypes.c_void_p,
    ctypes.c_size_t,
    ctypes.c_void_p,
]


def cudnnMultiHeadAttnForward(
    handle: int,
    attn_desc: int,
    curr_idx: int,
    lo_win_idx: ctypes.c_void_p,
    hi_win_idx: ctypes.c_void_p,
    dev_seq_lengths_qo: ctypes.c_void_p,
    dev_seq_lengths_kv: ctypes.c_void_p,
    q_desc: int,
    queries: ctypes.c_void_p,
    residuals: ctypes.c_void_p,
    k_desc: int,
    keys: ctypes.c_void_p,
    v_desc: int,
    values: ctypes.c_void_p,
    o_desc: int,
    out: ctypes.c_void_p,
    weight_size_in_bytes: int,
    weights: ctypes.c_void_p,
    work_space_size_in_bytes: int,
    work_space: ctypes.c_void_p,
    reserve_space_size_in_bytes: int,
    reserve_space: ctypes.c_void_p,
) -> None:
    """The cudnnMultiHeadAttnForward function computes the forward responses of the multi-head attention layer.

    When reserveSpaceSizeInBytes=0 and reserveSpace=NULL, the function operates in the inference mode in which
    backward (gradient) functions are not invoked, otherwise, the training mode is assumed. In the training mode,
    the reserve space is used to pass intermediate results from cudnnMultiHeadAttnForward() to
    cudnnMultiHeadAttnBackwardData() and from cudnnMultiHeadAttnBackwardData() to cudnnMultiHeadAttnBackwardWeights().

    Parameters
    ----------
    handle : cudnnHandle
        The current cuDNN context handle.
    attn_desc : cudnnAttnDescriptor
        A previously initialized attention descriptor.
    curr_idx : int
        Time-step in queries to process. When the curr_idx argument is negative, all Q time-steps are processed.
        When currIdx is zero or positive, the forward response is computed for the selected time-step only. The latter
        input can be used in inference mode only, to process one time-step while updating the next attention window and
        Q, R, K, V inputs in-between calls.
    lo_win_idx[], hi_win_idx[] : int[]
        Two host integer arrays specifying the start and end indices of the attention window for each Q time-step.
        The start index in K, V sets is inclusive, and the end index is exclusive.
    dev_seq_lengths_qo[] : int[]
        Device array specifying sequence lengths of query, residual, and output sequence data.
    dev_seq_lengths_kv[] : int[]
        Device array specifying sequence lengths of key and value input data.
    q_desc : cudnnSeqDataDescriptor
        Descriptor for the query and residual sequence data.
    queries : void_p
        Pointer to queries data in the device memory.
    residuals : void_p
        Pointer to residual data in device memory. Set this argument to NULL if no residual connections are
        required.
    k_desc : cudnnSeqDataDescriptor
        Descriptor for the keys sequence data.
    keys : void_p
        Pointer to keys data in device memory.
    v_desc : cudnnSeqDataDescriptor
        Descriptor for the values sequence data.
    values : void_p
        Pointer to values data in device memory.
    o_desc : cudnnSeqDataDescriptor
        Descriptor for the multi-head attention output sequence data.
    out : void_p
        Pointer to device memory where the output response should be written.
    weight_size_in_bytes : int
        Size of the weight buffer in bytes where all multi-head attention trainable parameters are stored.
    weights : void_p
        Pointer to the weight buffer in device memory.
    work_space_size_in_bytes : int
        Size of the work-space buffer in bytes used for temporary API storage.
    work_space : void_p
        Pointer to the work-space buffer in device memory.
    reserve_space_size_in_bytes : int
        Size of the reserve-space buffer in bytes used for data exchange between forward and backward (gradient)
        API calls. This parameter should be zero in the inference mode and non-zero in the training mode.
    reserve_space : void_p
        Pointer to the reserve-space buffer in device memory. This argument should be NULL in inference mode
        and non-NULL in the training mode.
    """
    assert _libcudnn
    status = _libcudnn.cudnnMultiHeadAttnForward(
        handle,
        attn_desc,
        curr_idx,
        lo_win_idx,
        hi_win_idx,
        dev_seq_lengths_qo,
        dev_seq_lengths_kv,
        q_desc,
        queries,
        residuals,
        k_desc,
        keys,
        v_desc,
        values,
        o_desc,
        out,
        weight_size_in_bytes,
        weights,
        work_space_size_in_bytes,
        work_space,
        reserve_space_size_in_bytes,
        reserve_space,
    )
    cudnnCheckStatus(status)


_libcudnn.cudnnMultiHeadAttnBackwardData.restype = int
_libcudnn.cudnnMultiHeadAttnBackwardData.argtypes = [
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_size_t,
    ctypes.c_void_p,
    ctypes.c_size_t,
    ctypes.c_void_p,
    ctypes.c_size_t,
    ctypes.c_void_p,
]


def cudnnMultiHeadAttnBackwardData(
    handle: int,
    attn_desc: int,
    lo_win_idx: ctypes.c_void_p,
    hi_win_idx: ctypes.c_void_p,
    dev_seq_lengths_dqdo: ctypes.c_void_p,
    dev_seq_lengths_dkdv: ctypes.c_void_p,
    do_desc: int,
    dout: ctypes.c_void_p,
    dq_desc: int,
    dqueries: ctypes.c_void_p,
    queries: ctypes.c_void_p,
    dk_desc: int,
    dkeys: ctypes.c_void_p,
    keys: ctypes.c_void_p,
    dv_desc: int,
    dvalues: ctypes.c_void_p,
    values: ctypes.c_void_p,
    weight_size_in_bytes: int,
    weights: ctypes.c_void_p,
    work_space_size_in_bytes: int,
    work_space: ctypes.c_void_p,
    reserve_space_size_in_bytes: int,
    reserve_space: ctypes.c_void_p,
) -> None:
    """This function computes exact, first-order derivatives of the multi-head attention block with respect to its inputs: Q, K, V.

    If y=F(x) is a vector-valued function that represents the multi-head attention layer and it takes some vector
    x ϵ ℝ n as an input (with all other parameters and inputs constant), and outputs vector y ϵ ℝ m , then
    cudnnMultiHeadAttnBackwardData() computes the result of ∂ y i / ∂ x j T δ out where δ out is the m × 1 gradient of the
    loss function with respect to multi-head attention outputs. The δ out gradient is back propagated through prior layers
    of the deep learning model. ∂ y i / ∂ x j is the m × n Jacobian matrix of F(x). The input is supplied via the dout argument
    and gradient results for Q, K, V are written to the dqueries, dkeys, and dvalues buffers.
    Parameters
    ----------
    handle : cudnnHandle
        The current cuDNN context handle.
    attn_desc : cudnnAttnDescriptor
        A previously initialized attention descriptor.
    lo_win_idx[], hi_wini_dx[] : int[]
        Two host integer arrays specifying the start and end indices of the attention window for each Q time-step.
        The start index in K, V sets is inclusive, and the end index is exclusive.
    dev_seq_lengths_dqdO[]: int[]
        Device array containing a copy of the sequence length array from the dqDesc or doDesc sequence data descriptor.
    dev_seq_lengths_dkdv[]: int[]
        Device array containing a copy of the sequence length array from the dkDesc or dvDesc sequence data descriptor.
    do_desc: cudnnSeqDataDescriptor
        Descriptor for the δ out gradients (vectors of partial derivatives of the loss function with respect to the multi-head attention outputs).
    dout : void_p
        Pointer to δ out gradient data in the device memory.
    dq_desc : int
        Descriptor for queries and dqueries sequence data.
    dqueries : void_p
        Device pointer to gradients of the loss function computed with respect to queries vectors.
    queries : void_p
        Pointer to queries data in the device memory. This is the same input as in cudnnMultiHeadAttnForward().
    dk_desc : int
        Descriptor for keys and dkeys sequence data.
    dkeys : void_p
        Device pointer to gradients of the loss function computed with respect to keys vectors.
    keys : void_p
        Pointer to keys data in the device memory. This is the same input as in cudnnMultiHeadAttnForward().
    dv_desc : int
        Descriptor for values and dvalues sequence data.
    dvalues : void_p
        Device pointer to gradients of the loss function computed with respect to values vectors.
    values : void_p
        Pointer to values data in the device memory. This is the same input as in cudnnMultiHeadAttnForward().
    weight_size_in_bytes
        Size of the weight buffer in bytes where all multi-head attention trainable parameters are stored.
    weights : void_p
        Pointer to the weight buffer in device memory.
    work_space_size_in_bytes
        Size of the work-space buffer in bytes used for temporary API storage.
    work_space : void_p
        Pointer to the work-space buffer in device memory.
    reserve_space_size_in_bytes : int
        Size of the reserve-space buffer in bytes used for data exchange between forward and backward (gradient)
        API calls. This parameter should be zero in the inference mode and non-zero in the training mode.
    reserve_space : void_p
        Pointer to the reserve-space buffer in device memory. This argument should be NULL in inference mode
        and non-NULL in the training mode.
    """

    assert _libcudnn
    status = _libcudnn.cudnnMultiHeadAttnBackwardData(
        handle,
        attn_desc,
        lo_win_idx,
        hi_win_idx,
        dev_seq_lengths_dqdo,
        dev_seq_lengths_dkdv,
        do_desc,
        dout,
        dq_desc,
        dqueries,
        queries,
        dk_desc,
        dkeys,
        keys,
        dv_desc,
        dvalues,
        values,
        weight_size_in_bytes,
        weights,
        work_space_size_in_bytes,
        work_space,
        reserve_space_size_in_bytes,
        reserve_space,
    )
    cudnnCheckStatus(status)


_libcudnn.cudnnMultiHeadAttnBackwardWeights.restype = int
_libcudnn.cudnnMultiHeadAttnBackwardWeights.argtypes = [
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_size_t,
    ctypes.c_void_p,
    ctypes.c_size_t,
    ctypes.c_void_p,
]


def cudnnMultiHeadAttnBackwardWeights(
    handle: int,
    attn_desc: int,
    add_grad: int,
    q_desc: int,
    queries: ctypes.c_void_p,
    k_desc: int,
    keys: ctypes.c_void_p,
    v_desc: int,
    values: ctypes.c_void_p,
    do_desc: int,
    dout: ctypes.c_void_p,
    weight_size_in_bytes: int,
    weights: ctypes.c_void_p,
    dweights: ctypes.c_void_p,
    work_space_size_in_bytes: int,
    work_space: ctypes.c_void_p,
    reserve_space_size_in_bytes: int,
    reserve_space: ctypes.c_void_p,
) -> None:
    """This function computes exact, first-order derivatives of the multi-head attention block with respect to its trainable parameters: projection weights and projection biases.

    If y=F(w) is a vector-valued function that represents the multi-head attention layer and it takes some vector
    w ϵ ℝⁿ of flattened weights or biases as an input (with all other parameters and inputs fixed), and outputs
    vector y ϵ ℝᵐ, then cudnnMultiHeadAttnBackwardWeights() computes the result of
    ∂yᵢ/∂wⱼᵀ δout, where δout is the m × 1 gradient of the loss function with respect to the multi-head attention
    outputs. The δout gradient is back propagated through prior layers of the deep learning model.
    ∂yᵢ/∂wⱼ is the m × n Jacobian matrix of F(w). The δout input is supplied via the dout argument.

    Parameters
    ----------
    handle : cudnnHandle
        The current cuDNN context handle.
    attn_desc : cudnnAttnDescriptor
        A previously initialized attention descriptor.
    add_grad : cudnnWgradMode
        Weight gradient accumulation mode.
    q_desc : cudnnSeqDataDescriptor
        Descriptor for queries sequence data.
    queries : void_p
        Pointer to queries data in the device memory.
    k_desc : cudnnSeqDataDescriptor
        Descriptor for keys sequence data.
    keys : void_p
        Pointer to keys data in the device memory.
    v_desc : cudnnSeqDataDescriptor
        Descriptor for values sequence data.
    values : void_p
        Pointer to values data in the device memory.
    do_desc : cudnnSeqDataDescriptor
        Descriptor for the δout gradients (vectors of partial derivatives of the loss function with respect to the
        multi-head attention outputs).
    dout : void_p
        Pointer to δout gradient data in the device memory.
    weight_size_in_bytes : int
        Size of the weight buffer in bytes where all multi-head attention trainable parameters are stored.
    weights : void_p
        Pointer to the weight buffer in device memory.
    dweights : void_p
        Pointer to the weight gradient buffer in device memory.
    work_space_size_in_bytes : int
        Size of the work-space buffer in bytes used for temporary API storage.
    work_space : void_p
        Pointer to the work-space buffer in device memory.
    reserve_space_size_in_bytes : int
        Size of the reserve-space buffer in bytes used for data exchange between forward and backward (gradient)
        API calls.
    reserve_space : void_p
        Pointer to the reserve-space buffer in device memory.
    """

    assert _libcudnn
    status = _libcudnn.cudnnMultiHeadAttnBackwardWeights(
        handle,
        attn_desc,
        add_grad,
        q_desc,
        queries,
        k_desc,
        keys,
        v_desc,
        values,
        do_desc,
        dout,
        weight_size_in_bytes,
        weights,
        dweights,
        work_space_size_in_bytes,
        work_space,
        reserve_space_size_in_bytes,
        reserve_space,
    )
    cudnnCheckStatus(status)


cudnnNormOps = {
    "CUDNN_NORM_OPS_NORM": 0,
    "CUDNN_NORM_OPS_NORM_ACTIVATION": 1,
    "CUDNN_NORM_OPS_NORM_ADD_ACTIVATION": 2,
}

cudnnNormAlgo = {"CUDNN_NORM_ALGO_STANDARD": 0, "CUDNN_NORM_ALGO_PERSIST": 1}

cudnnNormMode = {"CUDNN_NORM_PER_ACTIVATION": 0, "CUDNN_NORM_PER_CHANNEL": 1}

_libcudnn.cudnnNormalizationForwardTraining.restype = int
_libcudnn.cudnnNormalizationForwardTraining.argtypes = [
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_double,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_double,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_size_t,
    ctypes.c_void_p,
    ctypes.c_size_t,
    ctypes.c_int,
]


def cudnnNormalizationForwardTraining(
    handle: int,
    mode: int,
    norm_ops: int,
    algo: int,
    alpha: float,
    beta: float,
    x_desc: int,
    x_data: ctypes.c_void_p,
    norm_scale_bias_desc: int,
    norm_scale: ctypes.c_void_p,
    norm_bias: ctypes.c_void_p,
    exponential_average_factor: float,
    norm_mean_var_desc: int,
    result_running_mean: ctypes.c_void_p,
    result_running_variance: ctypes.c_void_p,
    epsilon: float,
    result_save_mean: ctypes.c_void_p,
    result_save_inv_variance: ctypes.c_void_p,
    activation_desc: int,
    z_desc: int,
    z_data: ctypes.c_void_p,
    y_desc: int,
    y_data: ctypes.c_void_p,
    work_space: ctypes.c_void_p,
    work_space_size_in_bytes: int,
    reserve_space: ctypes.c_void_p,
    reserve_space_size_in_bytes: int,
    group_cnt: int,
) -> None:
    """This function performs the forward normalization layer computation for the training phase.

    Depending on mode, different normalization operations will be performed.

    Parameters
    ----------
    handle : cudnnHandle
        Handle to a previously created cuDNN library descriptor.
    mode : cudnnNormMode
        Mode of operation (per-channel or per-activation).
    norm_ops : cudnnNormOps
        Mode of post-operation. This input can be used to perform either only the normalization,
        normalization followed by activation, or normalization followed by element-wise addition
        and then activation.
    algo : cudnnNormAlgo
        Normalization algorithm.
    alpha, beta : float
        Scaling factors used to blend the layer output value with the previous destination value:

            dst = alpha * result + beta * prior_dst

    x_desc : cudnnTensorDescriptor
        Descriptor for the input tensor.
    x_data : void_p
        Pointer to the input tensor data in device memory.
    norm_scale_bias_desc : cudnnTensorDescriptor
        Descriptor for the normalization scale and bias tensors.
    norm_scale : void_p
        Pointer to the normalization scale (gamma) in device memory.
    norm_bias : void_p
        Pointer to the normalization bias (beta) in device memory.
    exponential_average_factor : float
        Factor used for updating the running mean and variance.
    norm_mean_var_desc : cudnnTensorDescriptor
        Descriptor for the running and saved mean/variance tensors.
    result_running_mean : void_p
        Pointer to the running mean tensor in device memory.
    result_running_variance : void_p
        Pointer to the running variance tensor in device memory.
    epsilon : float
        Epsilon value used in the normalization formula.
    result_save_mean : void_p
        Pointer to the saved mean tensor used during the backward pass.
    result_save_inv_variance : void_p
        Pointer to the saved inverse variance tensor used during the backward pass.
    activation_desc : cudnnActivationDescriptor
        Activation descriptor used when activation is enabled.
    z_desc : cudnnTensorDescriptor
        Descriptor for the residual input tensor.
    z_data : void_p
        Pointer to the residual input tensor in device memory.
    y_desc : cudnnTensorDescriptor
        Descriptor for the output tensor.
    y_data : void_p
        Pointer to the output tensor in device memory.
    work_space : void_p
        Pointer to the workspace buffer in device memory.
    work_space_size_in_bytes : int
        Size of the workspace buffer in bytes.
    reserve_space : void_p
        Pointer to the reserve-space buffer in device memory.
    reserve_space_size_in_bytes : int
        Size of the reserve-space buffer in bytes.
    group_cnt : int
        Number of groups. Currently only the value 1 is supported.
    """

    data_type = cudnnGetTensor4dDescriptor(x_desc)[0]
    if data_type == cudnnDataType["CUDNN_DATA_DOUBLE"]:
        alpha_ref = ctypes.byref(ctypes.c_double(alpha))
        beta_ref = ctypes.byref(ctypes.c_double(beta))
    else:
        alpha_ref = ctypes.byref(ctypes.c_float(alpha))
        beta_ref = ctypes.byref(ctypes.c_float(beta))
    assert _libcudnn
    status = _libcudnn.cudnnNormalizationForwardTraining(
        handle,
        mode,
        norm_ops,
        algo,
        alpha_ref,
        beta_ref,
        x_desc,
        x_data,
        norm_scale_bias_desc,
        norm_scale,
        norm_bias,
        exponential_average_factor,
        norm_mean_var_desc,
        result_running_mean,
        result_running_variance,
        epsilon,
        result_save_mean,
        result_save_inv_variance,
        activation_desc,
        z_desc,
        z_data,
        y_desc,
        y_data,
        work_space,
        work_space_size_in_bytes,
        reserve_space,
        reserve_space_size_in_bytes,
        group_cnt,
    )
    cudnnCheckStatus(status)


_libcudnn.cudnnNormalizationForwardInference.restype = int
_libcudnn.cudnnNormalizationForwardInference.argtypes = [
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_double,
    ctypes.c_int,
]


def cudnnNormalizationForwardInference(
    handle: int,
    mode: int,
    norm_ops: int,
    algo: int,
    alpha: float,
    beta: float,
    x_desc: int,
    x: ctypes.c_void_p,
    norm_scale_bias_desc: int,
    norm_scale: ctypes.c_void_p,
    norm_bias: ctypes.c_void_p,
    norm_mean_var_desc: int,
    estimated_mean: ctypes.c_void_p,
    estimated_variance: ctypes.c_void_p,
    z_desc: int,
    z: ctypes.c_void_p,
    activation_desc: int,
    y_desc: int,
    y: ctypes.c_void_p,
    epsilon: float,
    group_cnt: int,
) -> None:
    """This function performs the forward normalization layer computation for the inference phase.

    Parameters
    ----------
    handle : cudnnHandle
        Handle to a previously created cuDNN library descriptor.
    mode : cudnnNormMode
        Mode of operation (per-channel or per-activation).
    norm_ops : cudnnNormOps
        Mode of post-operation. Currently only CUDNN_NORM_OPS_NORM is supported.
    algo : cudnnNormAlgo
        Normalization algorithm.
    alpha, beta : float
        Scaling factors used to blend the layer output value with the previous destination value:
        dst = alpha * result + beta * prior_dst
    x_desc : cudnnTensorDescriptor
        Descriptor for the input tensor.
    x : void_p
        Pointer to the input tensor data in device memory.
    norm_scale_bias_desc : cudnnTensorDescriptor
        Descriptor for the normalization scale and bias tensors.
    norm_scale : void_p
        Pointer to the normalization scale (gamma) in device memory.
    norm_bias : void_p
        Pointer to the normalization bias (beta) in device memory.
    norm_mean_var_desc : cudnnTensorDescriptor
        Descriptor for the mean and variance tensors.
    estimated_mean : void_p
        Pointer to the estimated mean tensor computed during training.
    estimated_variance : void_p
        Pointer to the estimated variance tensor computed during training.
    z_desc : cudnnTensorDescriptor
        Descriptor for the residual input tensor.
    z : void_p
        Pointer to the residual input tensor in device memory.
    activation_desc : cudnnActivationDescriptor
        Descriptor for the activation operation.
    y_desc : cudnnTensorDescriptor
        Descriptor for the output tensor.
    y : void_p
        Pointer to the output tensor in device memory.
    epsilon : float
        Epsilon value used in the normalization formula.
    group_cnt : int
        Number of groups. Currently only the value 1 is supported.
    """
    data_type = cudnnGetTensor4dDescriptor(x_desc)[0]
    if data_type == cudnnDataType["CUDNN_DATA_DOUBLE"]:
        alpha_ref = ctypes.byref(ctypes.c_double(alpha))
        beta_ref = ctypes.byref(ctypes.c_double(beta))
    else:
        alpha_ref = ctypes.byref(ctypes.c_float(alpha))
        beta_ref = ctypes.byref(ctypes.c_float(beta))
    assert _libcudnn
    status = _libcudnn.cudnnNormalizationForwardInference(
        handle,
        mode,
        norm_ops,
        algo,
        alpha_ref,
        beta_ref,
        x_desc,
        x,
        norm_scale_bias_desc,
        norm_scale,
        norm_bias,
        norm_mean_var_desc,
        estimated_mean,
        estimated_variance,
        z_desc,
        z,
        activation_desc,
        y_desc,
        y,
        epsilon,
        group_cnt,
    )
    cudnnCheckStatus(status)


_libcudnn.cudnnNormalizationBackward.restype = int
_libcudnn.cudnnNormalizationBackward.argtypes = [
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_double,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_size_t,
    ctypes.c_void_p,
    ctypes.c_size_t,
    ctypes.c_int,
]


def cudnnNormalizationBackward(
    handle: int,
    mode: int,
    norm_ops: int,
    algo: int,
    alpha_data_diff: float,
    beta_data_diff: float,
    alpha_param_diff: float,
    beta_param_diff: float,
    x_desc: int,
    x_data: ctypes.c_void_p,
    y_desc: int,
    y_data: ctypes.c_void_p,
    dy_desc: int,
    dy_data: ctypes.c_void_p,
    dz_desc: int,
    dz_data: ctypes.c_void_p,
    dx_desc: int,
    dx_data: ctypes.c_void_p,
    dnorm_scale_bias_desc: int,
    norm_scale_data: ctypes.c_void_p,
    norm_bias_data: ctypes.c_void_p,
    dnorm_scale_data: ctypes.c_void_p,
    dnorm_bias_data: ctypes.c_void_p,
    epsilon: float,
    norm_mean_var_desc: int,
    saved_mean: ctypes.c_void_p,
    saved_inv_variance: ctypes.c_void_p,
    activation_desc: int,
    work_space: ctypes.c_void_p,
    work_space_size_in_bytes: int,
    reserve_space: ctypes.c_void_p,
    reserve_space_size_in_bytes: int,
    group_cnt: int,
) -> None:
    """This function performs backward normalization layer computation.

    Parameters
    ----------
    handle : cudnnHandle
        Handle to a previously created cuDNN library descriptor.
    mode : cudnnNormMode
        Mode of operation (per-channel or per-activation).
    norm_ops : cudnnNormOps
        Mode of post-operation. This input can be used to perform either only the normalization,
        normalization followed by activation, or normalization followed by element-wise addition
        and then activation.
    algo : cudnnNormAlgo
        Normalization algorithm.
    alpha_data_diff, beta_data_diff : float
        Scaling factors used to blend the computed input gradient with the previous destination
        value:

            dst = alpha * result + beta * prior_dst

    alpha_param_diff, beta_param_diff : float
        Scaling factors used to blend the computed parameter gradients with the previous
        destination values:

            dst = alpha * result + beta * prior_dst

    x_desc : cudnnTensorDescriptor
        Descriptor for the input tensor.
    x_data : void_p
        Pointer to the input tensor in device memory.
    y_desc : cudnnTensorDescriptor
        Descriptor for the forward output tensor.
    y_data : void_p
        Pointer to the forward output tensor in device memory.
    dy_desc : cudnnTensorDescriptor
        Descriptor for the output gradient tensor.
    dy_data : void_p
        Pointer to the output gradient tensor in device memory.
    dz_desc : cudnnTensorDescriptor
        Descriptor for the residual gradient tensor.
    dz_data : void_p
        Pointer to the residual gradient tensor in device memory.
    dx_desc : cudnnTensorDescriptor
        Descriptor for the input gradient tensor.
    dx_data : void_p
        Pointer to the input gradient tensor in device memory.
    dnorm_scale_bias_desc : cudnnTensorDescriptor
        Shared descriptor for the normalization parameter tensors.
    norm_scale_data : void_p
        Pointer to the normalization scale (gamma) in device memory.
    norm_bias_data : void_p
        Pointer to the normalization bias (beta) in device memory.
    dnorm_scale_data : void_p
        Pointer to the normalization scale gradient in device memory.
    dnorm_bias_data : void_p
        Pointer to the normalization bias gradient in device memory.
    epsilon : float
        Epsilon value used in the normalization formula. The same value should be used during
        both the forward and backward passes.
    norm_mean_var_desc : cudnnTensorDescriptor
        Descriptor for the saved mean and inverse variance tensors.
    saved_mean : void_p
        Pointer to the saved mean tensor computed during the forward pass.
    saved_inv_variance : void_p
        Pointer to the saved inverse variance tensor computed during the forward pass.
    activation_desc : cudnnActivationDescriptor
        Descriptor for the activation operation.
    work_space : void_p
        Pointer to the workspace buffer in device memory.
    work_space_size_in_bytes : int
        Size of the workspace buffer in bytes.
    reserve_space : void_p
        Pointer to the reserve-space buffer in device memory.
    reserve_space_size_in_bytes : int
        Size of the reserve-space buffer in bytes.
    group_cnt : int
        Number of groups. Currently only the value 1 is supported.
    """
    data_type = cudnnGetTensor4dDescriptor(x_desc)[0]
    if data_type == cudnnDataType["CUDNN_DATA_DOUBLE"]:
        alpha_data_diff_ref = ctypes.byref(ctypes.c_double(alpha_data_diff))
        beta_data_diff_ref = ctypes.byref(ctypes.c_double(beta_data_diff))
        alpha_param_diff_ref = ctypes.byref(ctypes.c_double(alpha_param_diff))
        beta_param_diff_ref = ctypes.byref(ctypes.c_double(beta_param_diff))
    else:
        alpha_data_diff_ref = ctypes.byref(ctypes.c_float(alpha_data_diff))
        beta_data_diff_ref = ctypes.byref(ctypes.c_float(beta_data_diff))
        alpha_param_diff_ref = ctypes.byref(ctypes.c_float(alpha_param_diff))
        beta_param_diff_ref = ctypes.byref(ctypes.c_float(beta_param_diff))
    assert _libcudnn
    status = _libcudnn.cudnnNormalizationBackward(
        handle,
        mode,
        norm_ops,
        algo,
        alpha_data_diff_ref,
        beta_data_diff_ref,
        alpha_param_diff_ref,
        beta_param_diff_ref,
        x_desc,
        x_data,
        y_desc,
        y_data,
        dy_desc,
        dy_data,
        dz_desc,
        dz_data,
        dx_desc,
        dx_data,
        dnorm_scale_bias_desc,
        norm_scale_data,
        norm_bias_data,
        dnorm_scale_data,
        dnorm_bias_data,
        epsilon,
        norm_mean_var_desc,
        saved_mean,
        saved_inv_variance,
        activation_desc,
        work_space,
        work_space_size_in_bytes,
        reserve_space,
        reserve_space_size_in_bytes,
        group_cnt,
    )
    cudnnCheckStatus(status)


_libcudnn.cudnnGetNormalizationBackwardWorkspaceSize.restype = int
_libcudnn.cudnnGetNormalizationBackwardWorkspaceSize.argtypes = [
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_int,
]


def cudnnGetNormalizationBackwardWorkspaceSize(
    handle: int,
    mode: int,
    norm_ops: int,
    algo: int,
    x_desc: int,
    y_desc: int,
    dy_desc: int,
    dz_desc: int,
    dx_desc: int,
    dnorm_scale_bias_desc: int,
    activation_desc: int,
    norm_mean_var_desc: int,
    group_cnt: int,
) -> int:
    """Returns the workspace size required for backward normalization.

    Parameters
    ----------
    handle : cudnnHandle
        Handle to a previously created cuDNN library descriptor.
    mode : cudnnNormMode
        Mode of operation (per-channel or per-activation).
    norm_ops : cudnnNormOps
        Mode of post-operation.
    algo : cudnnNormAlgo
        Normalization algorithm.
    x_desc : cudnnTensorDescriptor
        Descriptor for the input tensor.
    y_desc : cudnnTensorDescriptor
        Descriptor for the output tensor.
    dy_desc : cudnnTensorDescriptor
        Descriptor for the output gradient tensor.
    dz_desc : cudnnTensorDescriptor
        Descriptor for the residual gradient tensor.
    dx_desc : cudnnTensorDescriptor
        Descriptor for the input gradient tensor.
    dnorm_scale_bias_desc : cudnnTensorDescriptor
        Shared descriptor for the normalization parameter tensors and their gradients.
    activation_desc : cudnnActivationDescriptor
        Descriptor for the activation operation.
    norm_mean_var_desc : cudnnTensorDescriptor
        Shared descriptor for the saved mean and inverse variance tensors.
    group_cnt : int
        Number of groups. Currently only the value 1 is supported.

    Returns
    -------
    ctypes.c_size_t
        Workspace size in bytes.
    """
    size_in_bytes = ctypes.c_size_t()
    assert _libcudnn
    status = _libcudnn.cudnnGetNormalizationBackwardWorkspaceSize(
        handle,
        mode,
        norm_ops,
        algo,
        x_desc,
        y_desc,
        dy_desc,
        dz_desc,
        dx_desc,
        dnorm_scale_bias_desc,
        activation_desc,
        norm_mean_var_desc,
        ctypes.byref(size_in_bytes),
        group_cnt,
    )
    cudnnCheckStatus(status)
    return size_in_bytes.value


_libcudnn.cudnnGetNormalizationForwardTrainingWorkspaceSize.restype = int
_libcudnn.cudnnGetNormalizationForwardTrainingWorkspaceSize.argtypes = [
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_int,
]


def cudnnGetNormalizationForwardTrainingWorkspaceSize(
    handle: int,
    mode: int,
    norm_ops: int,
    algo: int,
    x_desc: int,
    z_desc: int,
    y_desc: int,
    norm_scale_bias_desc: int,
    activation_desc: int,
    norm_mean_var_desc: int,
    group_cnt: int,
) -> int:
    """Returns the workspace size required for forward training normalization.

    Parameters
    ----------
    handle : cudnnHandle
        Handle to a previously created cuDNN library descriptor.
    mode : cudnnNormMode
        Mode of operation (per-channel or per-activation).
    norm_ops : cudnnNormOps
        Mode of post-operation.
    algo : cudnnNormAlgo
        Normalization algorithm.
    x_desc : cudnnTensorDescriptor
        Descriptor for the input tensor.
    z_desc : cudnnTensorDescriptor
        Descriptor for the residual input tensor.
    y_desc : cudnnTensorDescriptor
        Descriptor for the output tensor.
    norm_scale_bias_desc : cudnnTensorDescriptor
        Descriptor for the normalization scale and bias tensors.
    activation_desc : cudnnActivationDescriptor
        Descriptor for the activation operation.
    norm_mean_var_desc : cudnnTensorDescriptor
        Shared descriptor for the running mean and variance tensors.
    group_cnt : int
        Number of groups. Currently only the value 1 is supported.

    Returns
    -------
    ctypes.c_size_t
        Workspace size in bytes.
    """
    size_in_bytes = ctypes.c_size_t()
    assert _libcudnn
    status = _libcudnn.cudnnGetNormalizationForwardTrainingWorkspaceSize(
        handle,
        mode,
        norm_ops,
        algo,
        x_desc,
        z_desc,
        y_desc,
        norm_scale_bias_desc,
        activation_desc,
        norm_mean_var_desc,
        ctypes.byref(size_in_bytes),
        group_cnt,
    )
    cudnnCheckStatus(status)
    return size_in_bytes.value


_libcudnn.cudnnGetNormalizationTrainingReserveSpaceSize.restype = int
_libcudnn.cudnnGetNormalizationTrainingReserveSpaceSize.argtypes = [
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_int,
]


def cudnnGetNormalizationTrainingReserveSpaceSize(
    handle: int,
    mode: int,
    norm_ops: int,
    algo: int,
    activation_desc: int,
    x_desc: int,
    group_cnt: int,
) -> int:
    """Returns the reserve-space size required for training normalization.

    Parameters
    ----------
    handle : cudnnHandle
        Handle to a previously created cuDNN library descriptor.
    mode : cudnnNormMode
        Mode of operation (per-channel or per-activation).
    norm_ops : cudnnNormOps
        Mode of post-operation.
    algo : cudnnNormAlgo
        Normalization algorithm.
    activation_desc : cudnnActivationDescriptor
        Descriptor for the activation operation.
    x_desc : cudnnTensorDescriptor
        Descriptor for the input tensor.
    group_cnt : int
        Number of groups. Currently only the value 1 is supported.

    Returns
    -------
    ctypes.c_size_t
        Reserve-space size in bytes.
    """

    size_in_bytes = ctypes.c_size_t()

    assert _libcudnn
    status = _libcudnn.cudnnGetNormalizationTrainingReserveSpaceSize(
        handle,
        mode,
        norm_ops,
        algo,
        activation_desc,
        x_desc,
        ctypes.byref(size_in_bytes),
        group_cnt,
    )
    cudnnCheckStatus(status)
    return size_in_bytes.value


_libcudnn.cudnnBatchNormalizationBackward.restype = int
_libcudnn.cudnnBatchNormalizationBackward.argtypes = [
    ctypes.c_void_p,
    ctypes.c_int,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_double,
    ctypes.c_void_p,
    ctypes.c_void_p,
]


def cudnnBatchNormalizationBackward(
    handle: int,
    mode: int,
    alpha_data_diff: float,
    beta_data_diff: float,
    alpha_param_diff: float,
    beta_param_diff: float,
    x_desc: int,
    x: ctypes.c_void_p,
    dy_desc: int,
    dy: ctypes.c_void_p,
    dx_desc: int,
    dx: ctypes.c_void_p,
    bn_scale_bias_diff_desc: int,
    bn_scale: ctypes.c_void_p,
    result_bn_scale_diff: ctypes.c_void_p,
    result_bn_bias_diff: ctypes.c_void_p,
    epsilon: float,
    saved_mean: ctypes.c_void_p,
    saved_inv_variance: ctypes.c_void_p,
) -> None:
    """This function performs the backward batch normalization computation.

    This layer is based on the paper *Batch Normalization: Accelerating Deep Network Training
    by Reducing Internal Covariate Shift*, S. Ioffe and C. Szegedy, 2015.

    The same epsilon value must be used during training, backpropagation, and inference.

    Parameters
    ----------
    handle : cudnnHandle
        Handle to a previously created cuDNN library descriptor.
    mode : cudnnBatchNormMode
        Batch normalization mode (spatial or per-activation).
    alpha_data_diff, beta_data_diff : float
        Scaling factors used to blend the computed input gradient with the previous destination
        value:

            dst = alpha * result + beta * prior_dst

    alpha_param_diff, beta_param_diff : float
        Scaling factors used to blend the computed parameter gradients with the previous
        destination values:

            dst = alpha * result + beta * prior_dst

    x_desc : cudnnTensorDescriptor
        Descriptor for the input tensor.
    x : void_p
        Pointer to the input tensor in device memory.
    dy_desc : cudnnTensorDescriptor
        Descriptor for the output gradient tensor.
    dy : void_p
        Pointer to the output gradient tensor in device memory.
    dx_desc : cudnnTensorDescriptor
        Descriptor for the input gradient tensor.
    dx : void_p
        Pointer to the input gradient tensor in device memory.
    bn_scale_bias_diff_desc : cudnnTensorDescriptor
        Shared descriptor for the batch normalization parameter tensors and their gradients.
    bn_scale : void_p
        Pointer to the batch normalization scale (gamma) in device memory.
    result_bn_scale_diff : void_p
        Pointer to the computed scale gradient in device memory.
    result_bn_bias_diff : void_p
        Pointer to the computed bias gradient in device memory.
    epsilon : float
        Epsilon value used in the batch normalization formula. The same value should be used
        during both the forward and backward passes.
    saved_mean : void_p
        Pointer to the saved mean tensor computed during the forward pass.
    saved_inv_variance : void_p
        Pointer to the saved inverse variance tensor computed during the forward pass.
    """
    data_type = cudnnGetTensor4dDescriptor(x_desc)[0]
    if data_type == cudnnDataType["CUDNN_DATA_DOUBLE"]:
        alpha_data_diff_ref = ctypes.byref(ctypes.c_double(alpha_data_diff))
        beta_data_diff_ref = ctypes.byref(ctypes.c_double(beta_data_diff))
        alpha_param_diff_ref = ctypes.byref(ctypes.c_double(alpha_param_diff))
        beta_param_diff_ref = ctypes.byref(ctypes.c_double(beta_param_diff))
    else:
        alpha_data_diff_ref = ctypes.byref(ctypes.c_float(alpha_data_diff))
        beta_data_diff_ref = ctypes.byref(ctypes.c_float(beta_data_diff))
        alpha_param_diff_ref = ctypes.byref(ctypes.c_float(alpha_param_diff))
        beta_param_diff_ref = ctypes.byref(ctypes.c_float(beta_param_diff))
    assert _libcudnn
    status = _libcudnn.cudnnBatchNormalizationBackward(
        handle,
        mode,
        alpha_data_diff_ref,
        beta_data_diff_ref,
        alpha_param_diff_ref,
        beta_param_diff_ref,
        x_desc,
        x,
        dy_desc,
        dy,
        dx_desc,
        dx,
        bn_scale_bias_diff_desc,
        bn_scale,
        result_bn_scale_diff,
        result_bn_bias_diff,
        epsilon,
        saved_mean,
        saved_inv_variance,
    )
    cudnnCheckStatus(status)


_libcudnn.cudnnBatchNormalizationForwardInference.restype = int
_libcudnn.cudnnBatchNormalizationForwardInference.argtypes = [
    ctypes.c_void_p,
    ctypes.c_int,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_double,
]


def cudnnBatchNormalizationForwardInference(
    handle: int,
    mode: int,
    alpha: float,
    beta: float,
    x_desc: int,
    x: ctypes.c_void_p,
    y_desc: int,
    y: ctypes.c_void_p,
    bn_scale_bias_mean_var_desc: int,
    bn_scale: ctypes.c_void_p,
    bn_bias: ctypes.c_void_p,
    estimated_mean: ctypes.c_void_p,
    estimated_variance: ctypes.c_void_p,
    epsilon: float,
) -> None:
    """This function performs the forward batch normalization computation for the inference phase.

    This layer is based on the paper *Batch Normalization: Accelerating Deep Network Training
    by Reducing Internal Covariate Shift*, S. Ioffe and C. Szegedy, 2015.

    Parameters
    ----------
    handle : cudnnHandle
        Handle to a previously created cuDNN library descriptor.
    mode : cudnnBatchNormMode
        Batch normalization mode (spatial or per-activation).
    alpha, beta : float
        Scaling factors used to blend the layer output value with the previous destination
        value:

            dst = alpha * result + beta * prior_dst

    x_desc : cudnnTensorDescriptor
        Descriptor for the input tensor.
    x : void_p
        Pointer to the input tensor in device memory.
    y_desc : cudnnTensorDescriptor
        Descriptor for the output tensor.
    y : void_p
        Pointer to the output tensor in device memory.
    bn_scale_bias_mean_var_desc : cudnnTensorDescriptor
        Shared descriptor for the batch normalization scale, bias, mean, and variance tensors.
    bn_scale : void_p
        Pointer to the batch normalization scale (gamma) in device memory.
    bn_bias : void_p
        Pointer to the batch normalization bias (beta) in device memory.
    estimated_mean : void_p
        Pointer to the estimated mean tensor computed during training.
    estimated_variance : void_p
        Pointer to the estimated variance tensor computed during training.
    epsilon : float
        Epsilon value used in the batch normalization formula.
    """

    data_type = cudnnGetTensor4dDescriptor(x_desc)[0]
    if data_type == cudnnDataType["CUDNN_DATA_DOUBLE"]:
        alpha_ref = ctypes.byref(ctypes.c_double(alpha))
        beta_ref = ctypes.byref(ctypes.c_double(beta))
    else:
        alpha_ref = ctypes.byref(ctypes.c_float(alpha))
        beta_ref = ctypes.byref(ctypes.c_float(beta))
    assert _libcudnn
    status = _libcudnn.cudnnBatchNormalizationForwardInference(
        handle,
        mode,
        alpha_ref,
        beta_ref,
        x_desc,
        x,
        y_desc,
        y,
        bn_scale_bias_mean_var_desc,
        bn_scale,
        bn_bias,
        estimated_mean,
        estimated_variance,
        epsilon,
    )
    cudnnCheckStatus(status)


_libcudnn.cudnnBatchNormalizationForwardTraining.restype = int
_libcudnn.cudnnBatchNormalizationForwardTraining.argtypes = [
    ctypes.c_void_p,
    ctypes.c_int,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_double,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_double,
    ctypes.c_void_p,
    ctypes.c_void_p,
]


def cudnnBatchNormalizationForwardTraining(
    handle: int,
    mode: int,
    alpha: float,
    beta: float,
    x_desc: int,
    x: ctypes.c_void_p,
    y_desc: int,
    y: ctypes.c_void_p,
    bn_scale_bias_mean_var_desc: int,
    bn_scale: ctypes.c_void_p,
    bn_bias: ctypes.c_void_p,
    exponential_average_factor: float,
    result_running_mean: ctypes.c_void_p,
    result_running_variance: ctypes.c_void_p,
    epsilon: float,
    result_save_mean: ctypes.c_void_p,
    result_save_inv_variance: ctypes.c_void_p,
) -> None:
    """This function performs the forward batch normalization computation for the training phase.

    This layer is based on the paper *Batch Normalization: Accelerating Deep Network Training
    by Reducing Internal Covariate Shift*, S. Ioffe and C. Szegedy, 2015.

    Parameters
    ----------
    handle : cudnnHandle
        Handle to a previously created cuDNN library descriptor.
    mode : cudnnBatchNormMode
        Batch normalization mode (spatial or per-activation).
    alpha, beta : float
        Scaling factors used to blend the layer output value with the previous destination
        value:

            dst = alpha * result + beta * prior_dst

    x_desc : cudnnTensorDescriptor
        Descriptor for the input tensor.
    x : void_p
        Pointer to the input tensor in device memory.
    y_desc : cudnnTensorDescriptor
        Descriptor for the output tensor.
    y : void_p
        Pointer to the output tensor in device memory.
    bn_scale_bias_mean_var_desc : cudnnTensorDescriptor
        Shared descriptor for the batch normalization scale, bias, mean, and variance tensors.
    bn_scale : void_p
        Pointer to the batch normalization scale (gamma) in device memory.
    bn_bias : void_p
        Pointer to the batch normalization bias (beta) in device memory.
    exponential_average_factor : float
        Factor used to update the running mean and variance.
    result_running_mean : void_p
        Pointer to the running mean tensor in device memory.
    result_running_variance : void_p
        Pointer to the running variance tensor in device memory.
    epsilon : float
        Epsilon value used in the batch normalization formula. The same value should be used
        during both the forward and backward passes.
    result_save_mean : void_p
        Pointer to the saved mean tensor used during the backward pass.
    result_save_inv_variance : void_p
        Pointer to the saved inverse variance tensor used during the backward pass.
    """

    data_type = cudnnGetTensor4dDescriptor(x_desc)[0]
    if data_type == cudnnDataType["CUDNN_DATA_DOUBLE"]:
        alpha_ref = ctypes.byref(ctypes.c_double(alpha))
        beta_ref = ctypes.byref(ctypes.c_double(beta))
    else:
        alpha_ref = ctypes.byref(ctypes.c_float(alpha))
        beta_ref = ctypes.byref(ctypes.c_float(beta))
    assert _libcudnn
    status = _libcudnn.cudnnBatchNormalizationForwardTraining(
        handle,
        mode,
        alpha_ref,
        beta_ref,
        x_desc,
        x,
        y_desc,
        y,
        bn_scale_bias_mean_var_desc,
        bn_scale,
        bn_bias,
        exponential_average_factor,
        result_running_mean,
        result_running_variance,
        epsilon,
        result_save_mean,
        result_save_inv_variance,
    )
    cudnnCheckStatus(status)


_libcudnn.cudnnActivationForward.restype = int
_libcudnn.cudnnActivationForward.argtypes = [
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_void_p,
]


def cudnnActivationForward(
    handle: int,
    activation_desc: int,
    alpha: float,
    x_desc: int,
    x_data: ctypes.c_void_p,
    beta: float,
    y_desc: int,
    y_data: ctypes.c_void_p,
) -> None:
    """Applies an activation function element-wise to the input tensor.

    In-place operation is supported. In that case, ``x_data`` and ``y_data`` may point
    to the same memory location, provided that ``x_desc`` and ``y_desc`` are identical.

    Parameters
    ----------
    handle : cudnnHandle
        Handle to a previously created cuDNN context.
    activation_desc : cudnnActivationDescriptor
        Previously initialized activation descriptor.
    alpha : float
        Scaling factor applied to each element of the input tensor.
    x_desc : cudnnTensorDescriptor
        Descriptor for the input tensor.
    x_data : void_p
        Pointer to the input tensor in device memory.
    beta : float
        Scaling factor applied to the output tensor before adding the activation result.
        If zero, the output tensor is not read before writing the result.
    y_desc : cudnnTensorDescriptor
        Descriptor for the output tensor.
    y_data : void_p
        Pointer to the output tensor in device memory.
    """

    data_type = cudnnGetTensor4dDescriptor(y_desc)[0]
    if data_type == cudnnDataType["CUDNN_DATA_DOUBLE"]:
        alpha_ref = ctypes.byref(ctypes.c_double(alpha))
        beta_ref = ctypes.byref(ctypes.c_double(beta))
    else:
        alpha_ref = ctypes.byref(ctypes.c_float(alpha))
        beta_ref = ctypes.byref(ctypes.c_float(beta))

    assert _libcudnn
    status = _libcudnn.cudnnActivationForward(
        handle,
        activation_desc,
        alpha_ref,
        x_desc,
        x_data,
        beta_ref,
        y_desc,
        y_data,
    )
    cudnnCheckStatus(status)


_libcudnn.cudnnActivationBackward.restype = int
_libcudnn.cudnnActivationBackward.argtypes = [
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_void_p,
]


def cudnnActivationBackward(
    handle: int,
    activation_desc: int,
    alpha: float,
    y_desc: int,
    y_data: ctypes.c_void_p,
    dy_desc: int,
    dy_data: ctypes.c_void_p,
    x_desc: int,
    x_data: ctypes.c_void_p,
    beta: float,
    dx_desc: int,
    dx_data: ctypes.c_void_p,
) -> None:
    """Computes the gradient of an activation function.

    In-place operation is supported. In that case, the input and output tensors, as well as
    their corresponding gradient tensors, may share the same memory locations, provided that
    their tensor descriptors are identical.

    Parameters
    ----------
    handle : cudnnHandle
        Handle to a previously created cuDNN context.
    activation_desc : cudnnActivationDescriptor
        Previously initialized activation descriptor.
    alpha : float
        Scaling factor applied to the computed activation gradient.
    y_desc : cudnnTensorDescriptor
        Descriptor for the forward output tensor.
    y_data : void_p
        Pointer to the forward output tensor in device memory.
    dy_desc : cudnnTensorDescriptor
        Descriptor for the output gradient tensor.
    dy_data : void_p
        Pointer to the output gradient tensor in device memory.
    x_desc : cudnnTensorDescriptor
        Descriptor for the forward input tensor.
    x_data : void_p
        Pointer to the forward input tensor in device memory.
    beta : float
        Scaling factor applied to the destination gradient tensor before adding the computed
        activation gradient. If zero, the destination tensor is not read before writing.
    dx_desc : cudnnTensorDescriptor
        Descriptor for the input gradient tensor.
    dx_data : void_p
        Pointer to the input gradient tensor in device memory.
    """
    data_type = cudnnGetTensor4dDescriptor(x_desc)[0]
    if data_type == cudnnDataType["CUDNN_DATA_DOUBLE"]:
        alpha_ref = ctypes.byref(ctypes.c_double(alpha))
        beta_ref = ctypes.byref(ctypes.c_double(beta))
    else:
        alpha_ref = ctypes.byref(ctypes.c_float(alpha))
        beta_ref = ctypes.byref(ctypes.c_float(beta))
    assert _libcudnn
    status = _libcudnn.cudnnActivationBackward(
        handle,
        activation_desc,
        alpha_ref,
        y_desc,
        y_data,
        dy_desc,
        dy_data,
        x_desc,
        x_data,
        beta_ref,
        dx_desc,
        dx_data,
    )
    cudnnCheckStatus(status)
