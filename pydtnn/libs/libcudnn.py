"""
Python interface to the NVIDIA cuDNN library
"""

import sys
import ctypes
import ctypes.util

if sys.platform in ('linux2', 'linux'):
    _libcudnn_libname_list = ['libcudnn.so', 'libcudnn.so.7', 'libcudnn.so.6.0.21']
elif sys.platform == 'darwin':
    _libcudnn_libname_list = ['libcudnn.dylib', 'libcudnn.6.dylib']
elif sys.platform == 'win32':
    _libcudnn_libname_list = ['cudnn64_6.dll']
else:
    raise NotImplementedError('PyDTNN CUDNN: current platform is not yet supported!')

_libcudnn = None
for _libcudnn_libname in _libcudnn_libname_list:
    try:
        _libcudnn = ctypes.cdll.LoadLibrary(_libcudnn_libname)
    except OSError:
        pass
    else:
        break
if _libcudnn is None:
    raise OSError('cuDNN library not found')

# cuDNN error
_libcudnn.cudnnGetErrorString.restype = ctypes.c_char_p
_libcudnn.cudnnGetErrorString.argtypes = [ctypes.c_int]


class CudnnError(Exception):
    def __init__(self, status):
        self.status = status

    def __str__(self):
        error = _libcudnn.cudnnGetErrorString(self.status)
        return f'{error}'


# Data layout specification
# cudnnTensorFormat_t is an enumerated type used by
# cudnnSetTensor4dDescriptor() to create a tensor with a pre-defined layout.
type CudnnTensorFormat = dict[str, int]
cudnnTensorFormat = {
    'CUDNN_TENSOR_NCHW': 0,  # This tensor format specifies that the data
    # is laid out in the following order: image,
    # features map, rows, columns. The strides
    # are implicitly defined in such a way that
    # the data are contiguous in memory with no
    # padding between images, feature maps,
    # rows, and columns; the columns are the
    # inner dimension and the images are the
    # outermost dimension.
    'CUDNN_TENSOR_NHWC': 1,  # This tensor format specifies that the data
    # is laid out in the following order: image,
    # rows, columns, features maps. The strides
    # are implicitly defined in such a way that
    # the data are contiguous in memory with no
    # padding between images, rows, columns, and
    # features maps; the feature maps are the
    # inner dimension and the images are the
    # outermost dimension.
    'CUDNN_TENSOR_NCHW_VECT_C': 2  # This tensor format specifies that the data
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
    'CUDNN_DATA_FLOAT': 0,  # The data is 32-bit single-precision floating point
    # ( float ).
    'CUDNN_DATA_DOUBLE': 1,  # The data is 64-bit double-precision floating point
    # ( double ).
    'CUDNN_DATA_HALF': 2,  # The data is 16-bit half-precision floating point
    # ( half ).
    'CUDNN_DATA_INT8': 3,  # The data is 8-bit signed integer.
    'CUDNN_DATA_INT32': 4,  # The data is 32-bit signed integer.
    'CUDNN_DATA_INT8x4': 5  # The data is 32-bit element composed of 4 8-bit
    # signed integer. This data type is only supported
    # with tensor tensor_format CUDNN_TENSOR_NCHW_VECT_C.
}

# Math type
# cudnnMathType_t is an enumerated type used to indicate if the use of Tensor Core
# operations is permitted in a given library routine.
type CudnnMathType = dict[str, int]
cudnnMathType = {
    'CUDNN_DEFAULT_MATH': 0,  # Tensor Core operations are not used on
    # pre-NVIDIA A100 GPU devices. On A100 GPU architecture devices,
    # Tensor Core TF32 operation is permitted.
    'CUDNN_TENSOR_OP_MATH': 1,  # The use of Tensor Core operations is permitted
    # but will not actively perform datatype down conversion on tensors in order
    # to utilize Tensor Cores.
    'CUDNN_TENSOR_OP_MATH_ALLOW_CONVERSION': 2,  # The use of Tensor Core operations
    # is permitted and will actively perform datatype down conversion on tensors
    # in order to utilize Tensor Cores.
    'CUDNN_FMA_MATH': 3  # Restricted to only kernels that use FMA instructions.
}


# cudnnSeqDataAxis_t is an enumerated type used by cudnnSetSeqDataDescriptor()
# type cudnnSeqDataAxis = dict[str, int]
cudnnSeqDataAxis = {
    'CUDNN_SEQDATA_TIME_DIM':  0,  # Identifies the TIME (sequence length) dimension or
    #  specifies the TIME in the data layout.
    'CUDNN_SEQDATA_BATCH_DIM': 1,  # Identifies the BATCH dimension or specifies the BATCH
    # in the data layout.
    'CUDNN_SEQDATA_BEAM_DIM': 2,  # Identifies the BEAM dimension or specifies the BEAM in
    # the data layout.
    'CUDNN_SEQDATA_VECT_DIM': 3,  # Identifies the VECT (vector) dimension or specifies the
    # VECT in the data layout.
}

# type cudnnMultiHeadAttnWeightKind = dict[str, int]
cudnnMultiHeadAttnWeightKind = {
    'CUDNN_MH_ATTN_Q_WEIGHTS': 0,  # Selects the input projection weights for queries.
    'CUDNN_MH_ATTN_K_WEIGHTS': 1,  # Selects the input projection weights for keys.
    'CUDNN_MH_ATTN_V_WEIGHTS': 2,  # Selects the input projection weights for values.
    'CUDNN_MH_ATTN_O_WEIGHTS': 3,  # Selects the output projection weights.
    'CUDNN_MH_ATTN_Q_BIASES': 4,  # Selects the input projection biases for queries.
    'CUDNN_MH_ATTN_K_BIASES': 5,  # Selects the input projection biases for keys.
    'CUDNN_MH_ATTN_V_BIASES': 6,  # Selects the input projection biases for values.
    'CUDNN_MH_ATTN_O_BIASES': 7  # Selects the output projection biases.
}


# type cudnnAttnMode = dict[str, int]
cudnnAttnMode = {
    # Forward declaration of mapping between Q and K , V vectors when the beam size is greater than one in the Q input. Multiple Q vectors from the same beam bundle map to the same K , V vectors. This means that beam sizes in the K , V sets are equal to one.
    'CUDNN_ATTN_QUERYMAP_ALL_TO_ONE': 0,
    # Forward declaration of mapping between Q and K , V vectors when the beam size is greater than one in the Q input. Multiple Q vectors from the same beam bundle map to different K , V vectors. This requires beam sizes in K , V sets to be the same as in the Q input.
    'CUDNN_ATTN_QUERYMAP_ONE_TO_ONE': 1,
    'CUDNN_ATTN_DISABLE_PROJ_BIASES': 0,  # Use no biases in the attention input and output projections.
    # Use extra biases in the attention input and output projections. In this case the projected K ¯ vectors are computed as K i ¯ = W K , i K + b * 1 , 1 , ..., 1 1 × n , where n is the number of columns in the K matrix. In other words, the same column vector b is added to all columns of K after the weight matrix multiplication.
    'CUDNN_ATTN_ENABLE_PROJ_BIASES': 2
}


# type cudnnWgradMode = dict[str, int]
cudnnWgradMode = {
    # A weight gradient component corresponding to a new batch of inputs is added to previously evaluated weight gradients. Before using this mode, the buffer holding weight gradients should be initialized to zero. Alternatively, the first API call outputting to an uninitialized buffer should use the CUDNN_WGRAD_MODE_SET option.
    'CUDNN_WGRAD_MODE_ADD': 0,
    'CUDNN_WGRAD_MODE_SET': 1  # A weight gradient component, corresponding to a new batch of inputs, overwrites previously stored weight gradients in the output buffer.
}


# cudnnAddMode_t is an enumerated type used by cudnnAddTensor() to specify how
# a bias tensor is added to an input/output tensor.
type CudnnAddMode = dict[str, int]
cudnnAddMode = {
    'CUDNN_ADD_IMAGE': 0,
    'CUDNN_ADD_SAME_HW': 0,  # In this mode, the bias tensor is defined as one
    # image with one feature map. This image will be
    # added to every feature map of every image of the
    # input/output tensor.
    'CUDNN_ADD_FEATURE_MAP': 1,
    'CUDNN_ADD_SAME_CHW': 1,  # In this mode, the bias tensor is defined as one
    # image with multiple feature maps. This image
    # will be added to every image of the input/output
    # tensor.
    'CUDNN_ADD_SAME_C': 2,  # In this mode, the bias tensor is defined as one
    # image with multiple feature maps of dimension
    # 1x1; it can be seen as an vector of feature maps.
    # Each feature map of the bias tensor will be added
    # to the corresponding feature map of all height-by-
    # width pixels of every image of the input/output
    # tensor.
    'CUDNN_ADD_FULL_TENSOR': 3  # In this mode, the bias tensor has the same
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
    'CUDNN_CONVOLUTION': 0,  # In this mode, a convolution operation will be done
    # when applying the filter to the images.
    'CUDNN_CROSS_CORRELATION': 1  # In this mode, a cross-correlation operation will
    # be done when applying the filter to the images.
}

# cudnnConvolutionFwdPreference_t is an enumerated type used by
# cudnnGetConvolutionForwardAlgorithm() to help the choice of the algorithm used for the
# forward convolution.
type CudnnConvolutionFwdPreference = dict[str, int]
cudnnConvolutionFwdPreference = {
    'CUDNN_CONVOLUTION_FWD_NO_WORKSPACE': 0,  # In this configuration, the routine
    # cudnnGetConvolutionForwardAlgorithm() is guaranteed to return
    # an algorithm that does not require any extra workspace to be
    # provided by the user.
    'CUDNN_CONVOLUTION_FWD_PREFER_FASTEST': 1,  # In this configuration, the routine
    # cudnnGetConvolutionForwardAlgorithm() will return the fastest
    # algorithm regardless how much workspace is needed to execute it.
    'CUDNN_CONVOLUTION_FWD_SPECIFY_WORKSPACE_LIMIT': 2  # In this configuration, the routine
    # cudnnGetConvolutionForwardAlgorithm() will return the fastest
    # algorithm that fits within the memory limit that the user provided.
}

# cudnnConvolutionFwdAlgo_t is an enumerated type that exposes the different algorithm
# available to execute the forward convolution operation.
type CudnnConvolutionFwdAlgo = dict[str, int]
cudnnConvolutionFwdAlgo = {
    'CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM': 0,  # This algorithm expresses the convolution
    # as a matrix product without actually explicitly forming the matrix
    # that holds the input tensor data.
    'CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM': 1,  # This algorithm expresses the convolution
    # as a matrix product without actually explicitly forming the matrix
    # that holds the input tensor data, but still needs some memory
    # workspace to precompute some indices in order to facilitate the
    # implicit construction of the matrix that holds the input tensor data.
    'CUDNN_CONVOLUTION_FWD_ALGO_GEMM': 2,  # This algorithm expresses the convolution as an
    # explicit matrix product. A significant memory workspace is needed to
    # store the matrix that holds the input tensor data.
    'CUDNN_CONVOLUTION_FWD_ALGO_DIRECT': 3,  # This algorithm expresses the convolution as a
    # direct convolution (e.g without implicitly or explicitly doing a
    # matrix multiplication).
    'CUDNN_CONVOLUTION_FWD_ALGO_FFT': 4,
    'CUDNN_CONVOLUTION_FWD_ALGO_FFT_TILING': 5,
    'CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD': 6,
    'CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD_NONFUSED': 7,
    'CUDNN_CONVOLUTION_FWD_ALGO_COUNT': 8
}

type CudnnConvolutionBwdDataPreference = dict[str, int]
cudnnConvolutionBwdDataPreference = {
    'CUDNN_CONVOLUTION_BWD_DATA_NO_WORKSPACE': 0,
    'CUDNN_CONVOLUTION_BWD_DATA_PREFER_FASTEST': 1,
    'CUDNN_CONVOLUTION_BWD_DATA_SPECIFY_WORKSPACE_LIMIT': 2
}

type CudnnConvolutionBwdDataAlgo = dict[str, int]
cudnnConvolutionBwdDataAlgo = {
    'CUDNN_CONVOLUTION_BWD_DATA_ALGO_0': 0,
    'CUDNN_CONVOLUTION_BWD_DATA_ALGO_1': 1,
    'CUDNN_CONVOLUTION_BWD_DATA_ALGO_FFT': 2,
    'CUDNN_CONVOLUTION_BWD_DATA_ALGO_FFT_TILING': 3,
    'CUDNN_CONVOLUTION_BWD_DATA_ALGO_WINOGRAD': 4,
    'CUDNN_CONVOLUTION_BWD_DATA_ALGO_WINOGRAD_NONFUSED': 5,
    'CUDNN_CONVOLUTION_BWD_DATA_ALGO_COUNT': 6
}

type CudnnConvolutionBwdFilterPreference = dict[str, int]
cudnnConvolutionBwdFilterPreference = {
    'CUDNN_CONVOLUTION_BWD_FILTER_NO_WORKSPACE': 0,
    'CUDNN_CONVOLUTION_BWD_FILTER_PREFER_FASTEST': 1,
    'CUDNN_CONVOLUTION_BWD_FILTER_SPECIFY_WORKSPACE_LIMIT': 2,
}

type CudnnConvolutionBwdFilterAlgo = dict[str, int]
cudnnConvolutionBwdFilterAlgo = {
    'CUDNN_CONVOLUTION_BWD_FILTER_ALGO_0': 0,
    'CUDNN_CONVOLUTION_BWD_FILTER_ALGO_1': 1,
    'CUDNN_CONVOLUTION_BWD_FILTER_ALGO_FFT': 2,
    'CUDNN_CONVOLUTION_BWD_FILTER_ALGO_3': 3,
    'CUDNN_CONVOLUTION_BWD_FILTER_ALGO_WINOGRAD': 4,
    'CUDNN_CONVOLUTION_BWD_FILTER_ALGO_WINOGRAD_NONFUSED': 5,
    'CUDNN_CONVOLUTION_BWD_FILTER_ALGO_FFT_TILING': 6,
    'CUDNN_CONVOLUTION_BWD_FILTER_ALGO_COUNT': 7
}

type CudnnBatchNormMode = dict[str, int]
cudnnBatchNormMode = {
    'CUDNN_BATCHNORM_PER_ACTIVATION': 0,
    'CUDNN_BATCHNORM_SPATIAL': 1,
    'CUDNN_BATCHNORM_SPATIAL_PERSISTENT': 2
}

# cudnnSoftmaxAlgorithm_t is used to select an implementation of the softmax
# function used in cudnnSoftmaxForward() and cudnnSoftmaxBackward().
type CudnnSoftmaxAlgorithm = dict[str, int]
cudnnSoftmaxAlgorithm = {
    'CUDNN_SOFTMAX_FAST': 0,  # This implementation applies the straightforward
    # softmax operation.
    'CUDNN_SOFTMAX_ACCURATE': 1,  # This implementation applies a scaling to the input
    # to avoid any potential overflow.
    'CUDNN_SOFTMAX_LOG': 2  # This implementation applied the Log
    # softmax operation, scaling the input to avoid any potential
    # overflow.
}

# cudnnSoftmaxMode_t is used to select over which data the cudnnSoftmaxForward()
# and cudnnSoftmaxBackward() are computing their results.
type CudnnSoftmaxMode = dict[str, int]
cudnnSoftmaxMode = {
    'CUDNN_SOFTMAX_MODE_INSTANCE': 0,  # The softmax operation is computed per image (N)
    # across the dimensions C,H,W.
    'CUDNN_SOFTMAX_MODE_CHANNEL': 1  # The softmax operation is computed per spatial
    # location (H,W) per image (N) across the dimension
    # C.
}

# cudnnPoolingMode_t is an enumerated type passed to
# cudnnSetPoolingDescriptor() to select the pooling method to be used by
# cudnnPoolingForward() and cudnnPoolingBackward() .
type CudnnPoolingMode = dict[str, int]
cudnnPoolingMode = {
    'CUDNN_POOLING_MAX': 0,  # The maximum value inside the pooling window will
    # be used.
    'CUDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING': 1,  # The values inside the
    # pooling window will be averaged and this count
    # includes padded values.
    'CUDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING': 2,  # The values inside the
    #  pooling window will be averaged and this count
    # does not include padded values.
    'CUDNN_POOLING_MAX_DETERMINISTIC': 3  # The maximum value inside the pooling
    # window is used. The algorithm used is
    # deterministic.
}
# cudnnNanPropagation_t is an enumerated type used to indicate if a given routine
# should propagate Nan numbers. This enumerated type is used as a field for the
# cudnnActivationDescriptor_t descriptor and cudnnPoolingDescriptor_t descriptor
type CudnnNanPropagation = dict[str, int]
cudnnNanPropagation = {
    'CUDNN_NOT_PROPAGATE_NAN': 0,
    'CUDNN_PROPAGATE_NAN': 1
}
# cudnnActivationMode_t is an enumerated type used to select the neuron activation
# function used in cudnnActivationForward() and cudnnActivationBackward() .
type CudnnActivationMode = dict[str, int]
cudnnActivationMode = {
    'CUDNN_ACTIVATION_SIGMOID': 0,  # sigmoid function
    'CUDNN_ACTIVATION_RELU': 1,  # rectified linear function
    'CUDNN_ACTIVATION_TANH': 2,  # hyperbolic tangent function
    'CUDNN_ACTIVATION_CLIPPED_RELU': 3,
    'CUDNN_ACTIVATION_ELU': 4,
    'CUDNN_ACTIVATION_IDENTITY': 5
}


def cudnnCheckStatus(status):
    """
    Raise cuDNN exception
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


def cudnnGetVersion():
    """
    Get cuDNN Version.
    """
    return _libcudnn.cudnnGetVersion()


_libcudnn.cudnnCreate.restype = int
_libcudnn.cudnnCreate.argtypes = [ctypes.c_void_p]


def cudnnCreate():
    """
    Initialize cuDNN.
    Initializes cuDNN and returns a handle to the cuDNN context.
    Returns
    -------
    handle : cudnnHandle
        cuDNN context
    """

    handle = ctypes.c_void_p()
    status = _libcudnn.cudnnCreate(ctypes.byref(handle))
    cudnnCheckStatus(status)
    return handle.value


_libcudnn.cudnnDestroy.restype = int
_libcudnn.cudnnDestroy.argtypes = [ctypes.c_void_p]


def cudnnDestroy(handle):
    """
    Release cuDNN resources.
    Release hardware resources used by cuDNN.
    Parameters
    ----------
    handle : cudnnHandle
        cuDNN context.
    """

    status = _libcudnn.cudnnDestroy(ctypes.c_void_p(handle))
    cudnnCheckStatus(status)


_libcudnn.cudnnSetStream.restype = int
_libcudnn.cudnnSetStream.argtypes = [ctypes.c_void_p, ctypes.c_void_p]


def cudnnSetStream(handle, stream_id):
    """
    Set current cuDNN library stream.
    Parameters
    ----------
    handle : cudnnHandle
        cuDNN context.
    stream_id : cudaStream
        Stream Id.
    """

    status = _libcudnn.cudnnSetStream(handle, stream_id)
    cudnnCheckStatus(status)


_libcudnn.cudnnGetStream.restype = int
_libcudnn.cudnnGetStream.argtypes = [ctypes.c_void_p, ctypes.c_void_p]


def cudnnGetStream(handle):
    """
    Get current cuDNN library stream.
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
    status = _libcudnn.cudnnGetStream(handle, ctypes.byref(stream_id))
    cudnnCheckStatus(status)
    return stream_id.value


_libcudnn.cudnnCreateActivationDescriptor.restype = int
_libcudnn.cudnnCreateActivationDescriptor.argtypes = [ctypes.c_void_p]


def cudnnCreateActivationDescriptor():
    """
    Create a Activation descriptor object.
    Allocates a cudnnActivationDescriptor_t structure and returns a pointer to it.
    Returns
    -------
    Activation_descriptor : int
        Tensor descriptor.
    """

    activation = ctypes.c_void_p()
    status = _libcudnn.cudnnCreateActivationDescriptor(ctypes.byref(activation))
    cudnnCheckStatus(status)
    return activation.value


_libcudnn.cudnnSetActivationDescriptor.restype = int
_libcudnn.cudnnSetActivationDescriptor.argtypes = [ctypes.c_void_p, ctypes.c_int,
                                                   ctypes.c_int, ctypes.c_double]


def cudnnSetActivationDescriptor(activation_desc, mode, nan, coef):
    """
    Set a Activation descriptor object.
    Allocates a cudnnActivationDescriptor_t structure and returns a pointer to it.

    Parameters
    -----------
    activation_desc:  cudnnActivationDescriptor
        Handle to a previously created activation descriptor.
    nan: cudnnNanPropagation
        Enumerate to specify the nan propagation
    Returns
    -------
    Activation_descriptor : int
        Tensor descriptor.
    """

    status = _libcudnn.cudnnSetActivationDescriptor(activation_desc, mode, nan, coef)
    cudnnCheckStatus(status)


_libcudnn.cudnnCreateTensorDescriptor.restype = int
_libcudnn.cudnnCreateTensorDescriptor.argtypes = [ctypes.c_void_p]


def cudnnCreateTensorDescriptor():
    """
    Create a Tensor descriptor object.
    Allocates a cudnnTensorDescriptor_t structure and returns a pointer to it.
    Returns
    -------
    tensor_descriptor : int
        Tensor descriptor.
    """

    tensor = ctypes.c_void_p()
    status = _libcudnn.cudnnCreateTensorDescriptor(ctypes.byref(tensor))
    cudnnCheckStatus(status)
    return tensor.value


_libcudnn.cudnnSetTensor4dDescriptor.restype = int
_libcudnn.cudnnSetTensor4dDescriptor.argtypes = [ctypes.c_void_p, ctypes.c_int,
                                                 ctypes.c_int, ctypes.c_int,
                                                 ctypes.c_int, ctypes.c_int,
                                                 ctypes.c_int]


def cudnnSetTensor4dDescriptor(tensor_desc, tensor_format, data_type, n, c, h, w):
    """
    Initialize a previously created Tensor 4D object.
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

    status = _libcudnn.cudnnSetTensor4dDescriptor(tensor_desc, tensor_format, data_type,
                                                  n, c, h, w)
    cudnnCheckStatus(status)


_libcudnn.cudnnSetTensor4dDescriptorEx.restype = int
_libcudnn.cudnnSetTensor4dDescriptorEx.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int,
                                                   ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
                                                   ctypes.c_int, ctypes.c_int, ]


def cudnnSetTensor4dDescriptorEx(tensor_desc, data_type, n, c, h, w, n_stride, c_stride, h_stride, w_stride):
    """"
    Initialize a Tensor descriptor object with strides.
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

    status = _libcudnn.cudnnSetTensor4dDescriptorEx(tensor_desc, data_type, n, c, h, w,
                                                    n_stride, c_stride, h_stride, w_stride)
    cudnnCheckStatus(status)


_libcudnn.cudnnGetTensor4dDescriptor.restype = int
_libcudnn.cudnnGetTensor4dDescriptor.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                                                 ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                                                 ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                                                 ctypes.c_void_p, ]


def cudnnGetTensor4dDescriptor(tensor_desc):
    """"
    Get parameters of a Tensor descriptor object.
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

    status = _libcudnn.cudnnGetTensor4dDescriptor(tensor_desc, ctypes.byref(data_type), ctypes.byref(n),
                                                  ctypes.byref(c), ctypes.byref(h), ctypes.byref(w),
                                                  ctypes.byref(n_stride), ctypes.byref(c_stride),
                                                  ctypes.byref(h_stride), ctypes.byref(w_stride))
    cudnnCheckStatus(status)

    return (data_type.value, n.value, c.value, h.value, w.value, n_stride.value, c_stride.value,
            h_stride.value, w_stride.value)


_libcudnn.cudnnDestroyTensorDescriptor.restype = int
_libcudnn.cudnnDestroyTensorDescriptor.argtypes = [ctypes.c_void_p]


def cudnnDestroyTensorDescriptor(tensor_desc):
    """"
    Destroy a Tensor descriptor.
    This function destroys a previously created Tensor descriptor object.
    Parameters
    ----------
    tensor_desc : cudnnTensorDescriptor
        Previously allocated Tensor descriptor object.
    """

    status = _libcudnn.cudnnDestroyTensorDescriptor(tensor_desc)
    cudnnCheckStatus(status)


_libcudnn.cudnnTransformTensor.restype = int
_libcudnn.cudnnTransformTensor.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                                           ctypes.c_void_p, ctypes.c_void_p,
                                           ctypes.c_void_p, ctypes.c_void_p]


def cudnnTransformTensor(handle, alpha, src_desc, src_data, beta, dest_desc, dest_data):
    """"
    Tensor layout conversion helper (dest = alpha * src + beta * dest).
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
    src_desc : cudnnTensorDescriptor
        Handle to a previously initialized tensor descriptor.
    src_data : void_p
        Pointer to data of the tensor described by src_desc descriptor.
    beta: float
        Scaling factor which is applied on every element of the output tensor prior to adding
        the result of the operation. Note that if beta is zero, the output is not read and can
        contain any uninitialized data (including Nan numbers).
    dest_desc : cudnnTensorDescriptor
        Handle to a previously initialized tensor descriptor.
    dest_data : void_p
        Pointer to data of the tensor described by dest_desc descriptor.
    """

    data_type, _, _, _, _, _, _, _, _ = cudnnGetTensor4dDescriptor(dest_desc)
    if data_type == cudnnDataType['CUDNN_DATA_DOUBLE']:
        alpha_ref = ctypes.byref(ctypes.c_double(alpha))
        beta_ref = ctypes.byref(ctypes.c_double(beta))
    else:
        alpha_ref = ctypes.byref(ctypes.c_float(alpha))
        beta_ref = ctypes.byref(ctypes.c_float(beta))

    status = _libcudnn.cudnnTransformTensor(handle, alpha_ref, src_desc,
                                            src_data, beta_ref,
                                            dest_desc, dest_data)
    cudnnCheckStatus(status)


_libcudnn.cudnnAddTensor.restype = int
_libcudnn.cudnnAddTensor.argtypes = [ctypes.c_void_p, ctypes.c_void_p,
                                     ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                                     ctypes.c_void_p, ctypes.c_void_p]


def cudnnAddTensor(handle, alpha, bias_desc, bias_data, beta, src_dest_desc, src_dest_data):
    """"
    Tensor Bias addition : srcDest = alpha * bias + beta * src_dest_desc.
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
    if data_type == cudnnDataType['CUDNN_DATA_DOUBLE']:
        alpha_ref = ctypes.byref(ctypes.c_double(alpha))
        beta_ref = ctypes.byref(ctypes.c_double(beta))
    else:
        alpha_ref = ctypes.byref(ctypes.c_float(alpha))
        beta_ref = ctypes.byref(ctypes.c_float(beta))

    status = _libcudnn.cudnnAddTensor(handle, alpha_ref, bias_desc,
                                      bias_data, beta_ref,
                                      src_dest_desc, src_dest_data)
    cudnnCheckStatus(status)


_libcudnn.cudnnSetTensor.restype = int
_libcudnn.cudnnSetTensor.argtypes = [ctypes.c_void_p, ctypes.c_void_p,
                                     ctypes.c_void_p, ctypes.c_void_p]


def cudnnSetTensor(handle, src_desc, src_data, value):
    """"
    Set all data points of a tensor to a given value : srcDest = value.
    Parameters
    ----------
    handle : cudnnHandle
        Handle to a previously created cuDNN context.
    src_desc : cudnnTensorDescriptor
        Handle to a previously initialized tensor descriptor.
    src_data : void_p
        Pointer to data of the tensor described by src_desc descriptor.
    value : float
        Value that all elements of the tensor will be set to.
    """

    data_type, _, _, _, _, _, _, _, _ = cudnnGetTensor4dDescriptor(src_desc)
    if data_type == cudnnDataType['CUDNN_DATA_DOUBLE']:
        value_ref = ctypes.byref(ctypes.c_double(value))
    else:
        value_ref = ctypes.byref(ctypes.c_float(value))

    status = _libcudnn.cudnnSetTensor(handle, src_desc, src_data, value_ref)
    cudnnCheckStatus(status)


_libcudnn.cudnnScaleTensor.restype = int
_libcudnn.cudnnScaleTensor.argtypes = [ctypes.c_void_p, ctypes.c_void_p,
                                       ctypes.c_void_p, ctypes.c_void_p]


def cudnnScaleTensor(handle, src_desc, src_data, alpha):
    """"
    This function scales all the elements of a tensor by a give factor.
    Set all data points of a tensor to scaled value : srcDest = alpha * srcDest.
    Parameters
    ----------
    handle : cudnnHandle
        Handle to a previously created cuDNN context.
    src_desc : cudnnTensorDescriptor
        Handle to a previously initialized tensor descriptor.
    src_data : void_p
        Pointer to data of the tensor described by src_desc descriptor.
    alpha : float
        Value that all elements of the tensor will be scaled with.
    """

    data_type, _, _, _, _, _, _, _, _ = cudnnGetTensor4dDescriptor(src_desc)
    if data_type == cudnnDataType['CUDNN_DATA_DOUBLE']:
        alpha_ref = ctypes.byref(ctypes.c_double(alpha))
    else:
        alpha_ref = ctypes.byref(ctypes.c_float(alpha))

    status = _libcudnn.cudnnScaleTensor(handle, src_desc, src_data, alpha_ref)
    cudnnCheckStatus(status)


_libcudnn.cudnnCreateFilterDescriptor.restype = int
_libcudnn.cudnnCreateFilterDescriptor.argtypes = [ctypes.c_void_p]


def cudnnCreateFilterDescriptor():
    """"
    Create a filter descriptor.
    This function creates a filter descriptor object by allocating the memory needed
    to hold its opaque structure.
    Parameters
    ----------
    Returns
    -------
    w_desc : cudnnFilterDescriptor
        Handle to a newly allocated filter descriptor.
    """

    w_desc = ctypes.c_void_p()
    status = _libcudnn.cudnnCreateFilterDescriptor(ctypes.byref(w_desc))
    cudnnCheckStatus(status)

    return w_desc.value


_libcudnn.cudnnSetFilter4dDescriptor.restype = int
_libcudnn.cudnnSetFilter4dDescriptor.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int,
                                                 ctypes.c_int, ctypes.c_int, ctypes.c_int,
                                                 ctypes.c_int]


def cudnnSetFilter4dDescriptor(w_desc, data_type, tensor_format, k, c, h, w):
    """"
    Initialize a filter descriptor.
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

    status = _libcudnn.cudnnSetFilter4dDescriptor(w_desc, data_type, tensor_format, k, c, h, w)
    cudnnCheckStatus(status)


_libcudnn.cudnnGetFilter4dDescriptor.restype = int
_libcudnn.cudnnGetFilter4dDescriptor.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                                                 ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                                                 ctypes.c_void_p]


def cudnnGetFilter4dDescriptor(w_desc):
    """"
    Get parameters of filter descriptor.
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

    status = _libcudnn.cudnnGetFilter4dDescriptor(w_desc, ctypes.byref(data_type),
                                                  ctypes.byref(tensor_format),
                                                  ctypes.byref(k), ctypes.byref(c),
                                                  ctypes.byref(h), ctypes.byref(w))
    cudnnCheckStatus(status)

    return data_type.value, tensor_format.value, k.value, c.value, h.value, w.value


_libcudnn.cudnnDestroyFilterDescriptor.restype = int
_libcudnn.cudnnDestroyFilterDescriptor.argtypes = [ctypes.c_void_p]


def cudnnDestroyFilterDescriptor(w_desc):
    """"
    Destroy filter descriptor.
    This function destroys a previously created Tensor4D descriptor object.
    Parameters
    ----------
    w_desc : cudnnFilterDescriptor
    """

    status = _libcudnn.cudnnDestroyFilterDescriptor(w_desc)
    cudnnCheckStatus(status)


_libcudnn.cudnnCreateConvolutionDescriptor.restype = int
_libcudnn.cudnnCreateConvolutionDescriptor.argtypes = [ctypes.c_void_p]


def cudnnCreateConvolutionDescriptor():
    """"
    Create a convolution descriptor.
    This function creates a convolution descriptor object by allocating the memory needed to
    hold its opaque structure.
    Returns
    -------
    conv_desc : cudnnConvolutionDescriptor
        Handle to newly allocated convolution descriptor.
    """

    conv_desc = ctypes.c_void_p()

    status = _libcudnn.cudnnCreateConvolutionDescriptor(ctypes.byref(conv_desc))
    cudnnCheckStatus(status)

    return conv_desc.value


_libcudnn.cudnnSetConvolution2dDescriptor.restype = int
_libcudnn.cudnnSetConvolution2dDescriptor.argtypes = [ctypes.c_void_p, ctypes.c_int,
                                                      ctypes.c_int, ctypes.c_int, ctypes.c_int,
                                                      ctypes.c_int, ctypes.c_int, ctypes.c_int]


def cudnnSetConvolution2dDescriptor(conv_desc, pad_h, pad_w, u, v, dilation_h, dilation_w, mode,
                                    compute_type):
    """"
    Initialize a convolution descriptor.
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

    status = _libcudnn.cudnnSetConvolution2dDescriptor(conv_desc, pad_h, pad_w, u, v,
                                                       dilation_h, dilation_w, mode,
                                                       compute_type)
    cudnnCheckStatus(status)


_libcudnn.cudnnGetConvolution2dDescriptor.restype = int
_libcudnn.cudnnGetConvolution2dDescriptor.argtypes = [ctypes.c_void_p]


def cudnnGetConvolution2dDescriptor(conv_desc):
    """"
    Get a convolution descriptor.
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

    status = _libcudnn.cudnnGetConvolution2dDescriptor(conv_desc, ctypes.byref(pad_h),
                                                       ctypes.byref(pad_w), ctypes.byref(u),
                                                       ctypes.byref(v), ctypes.byref(dilation_h),
                                                       ctypes.byref(dilation_w),
                                                       ctypes.byref(mode), ctypes.byref(compute_type))

    cudnnCheckStatus(status)

    return (pad_h.value, pad_w.value, u.value, v.value, dilation_h.value, dilation_w.value, mode.value,
            compute_type.value)


_libcudnn.cudnnGetConvolution2dForwardOutputDim.restype = int
_libcudnn.cudnnGetConvolution2dForwardOutputDim.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                                                            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                                                            ctypes.c_void_p]


def cudnnGetConvolution2dForwardOutputDim(conv_desc, input_tensor_desc, w_desc):
    """"
    Return the dimensions of the output tensor given a convolution descriptor.
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

    status = _libcudnn.cudnnGetConvolution2dForwardOutputDim(conv_desc, input_tensor_desc,
                                                             w_desc, ctypes.byref(n),
                                                             ctypes.byref(c), ctypes.byref(h),
                                                             ctypes.byref(w))
    cudnnCheckStatus(status)

    return n.value, c.value, h.value, w.value


_libcudnn.cudnnSetConvolutionNdDescriptor.restype = int
_libcudnn.cudnnSetConvolutionNdDescriptor.argtypes = [ctypes.c_void_p,  # conv_desc
                                                      ctypes.c_int,  # arrayLength
                                                      ctypes.POINTER(ctypes.c_int),  # pad_a[]
                                                      ctypes.POINTER(ctypes.c_int),  # filter_stride_a[]
                                                      ctypes.POINTER(ctypes.c_int),  # dilation_a[]
                                                      ctypes.c_int,  # mode
                                                      ctypes.c_int]  # data_type


def cudnnSetConvolutionNdDescriptor(conv_desc, pad_a, filter_stride_a, dilation_a, mode, data_type):
    dim = len(pad_a)
    status = _libcudnn.cudnnSetConvolutionNdDescriptor(conv_desc,
                                                       dim,
                                                       (ctypes.c_int * dim)(*pad_a),
                                                       (ctypes.c_int * dim)(*filter_stride_a),
                                                       (ctypes.c_int * dim)(*dilation_a),
                                                       mode,
                                                       data_type)
    cudnnCheckStatus(status)


_libcudnn.cudnnDestroyConvolutionDescriptor.restype = int
_libcudnn.cudnnDestroyConvolutionDescriptor.argtypes = [ctypes.c_void_p]


def cudnnDestroyConvolutionDescriptor(conv_desc):
    """"
    Destroy a convolution descriptor.
    This function destroys a previously created convolution descriptor object.
    Parameters
    ----------
    conv_desc : int
        Previously created convolution descriptor.
    """

    status = _libcudnn.cudnnDestroyConvolutionDescriptor(conv_desc)
    cudnnCheckStatus(status)


class CudnnConvolutionFwdAlgoPerf(ctypes.Structure):
    _fields_ = [("algo", ctypes.c_int),
                ("status", ctypes.c_int),
                ("time", ctypes.c_float),
                ("memory", ctypes.c_size_t)]

    def __str__(self):
        return '(algo=%d, status=%d, time=%f, memory=%d)' % (self.algo,
                                                             self.status,
                                                             self.time,
                                                             self.memory)

    def __repr__(self):
        return self.__str__()


_libcudnn.cudnnFindConvolutionForwardAlgorithm.restype = int
_libcudnn.cudnnFindConvolutionForwardAlgorithm.argtypes = [ctypes.c_void_p,  # handle
                                                           ctypes.c_void_p,  # x_desc
                                                           ctypes.c_void_p,  # w_desc
                                                           ctypes.c_void_p,  # conv_desc
                                                           ctypes.c_void_p,  # y_desc
                                                           ctypes.c_int,  # requestAlgoCount
                                                           ctypes.c_void_p,  # returned_algo_count
                                                           ctypes.c_void_p]  # perf_results


def cudnnFindConvolutionForwardAlgorithm(handle, x_desc, w_desc, conv_desc, y_desc, requested_algo_count):
    perf_results_type = CudnnConvolutionFwdAlgoPerf * requested_algo_count
    perf_results = perf_results_type()
    returned_algo_count = ctypes.c_int()
    status = _libcudnn.cudnnFindConvolutionForwardAlgorithm(handle,
                                                            x_desc,
                                                            w_desc,
                                                            conv_desc,
                                                            y_desc,
                                                            ctypes.c_int(requested_algo_count),
                                                            ctypes.byref(returned_algo_count),
                                                            ctypes.cast(perf_results,
                                                                        ctypes.POINTER(CudnnConvolutionFwdAlgoPerf)))
    cudnnCheckStatus(status)
    return perf_results[0:returned_algo_count.value]


# _libcudnn.cudnnGetConvolutionForwardAlgorithm.restype = int
# _libcudnn.cudnnGetConvolutionForwardAlgorithm.argtypes = [ctypes.c_void_p,
#                                                          ctypes.c_void_p, ctypes.c_void_p,
#                                                          ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int,
#                                                          ctypes.c_size_t, ctypes.c_void_p]
# def cudnnGetConvolutionForwardAlgorithm(handle, src_desc, w_desc,
#                                        conv_desc, dest_desc, preference, memoryLimitInbytes):
#    """"
#    This function returns the best algorithm to choose for the forward convolution
#    depending on the criteria expressed in the cudnnConvolutionFwdPreference_t enumerant.
#    Parameters
#    ----------
#    handle : cudnnHandle
#        Handle to a previously created cuDNN context.
#    src_desc : cudnnTensorDescriptor
#        Handle to a previously initialized tensor descriptor.
#    w_desc : cudnnFilterDescriptor
#        Handle to a previously initialized filter descriptor.
#    conv_desc : cudnnConvolutionDescriptor
#        Previously initialized convolution descriptor.
#    dest_desc : cudnnTensorDescriptor
#        Handle to a previously initialized tensor descriptor.
#    preference : cudnnConvolutionFwdPreference
#        Enumerant to express the preference criteria in terms of memory
#        requirement and speed.
#    memoryLimitInbytes: size_t
#        The maximum amount of GPU memory the user is willing to use as a workspace
#        when preference is CUDNN_CONVOLUTION_FWD_SPECIFY_WORKSPACE_LIMIT.
#    Returns
#    -------
#    algo: cudnnConvolutionFwdAlgo
#        Enumerant that specifies which convolution algorithm should be used to
#        compute the results according to the specified preference.
#    """
#    algo = ctypes.c_int()
#
#    status = _libcudnn.cudnnGetConvolutionForwardAlgorithm(handle, src_desc, w_desc,
#                                               conv_desc, dest_desc, preference,
#                                               ctypes.c_size_t(memoryLimitInbytes),
#                                               ctypes.byref(algo))
#    cudnnCheckStatus(status)
#
#    return algo
#

_libcudnn.cudnnSetConvolutionGroupCount.restype = int
_libcudnn.cudnnSetConvolutionGroupCount.argtypes = [ctypes.c_void_p,
                                                    ctypes.c_int]


def cudnnSetConvolutionGroupCount(conv_desc, group_count):
    """"
    This function allows the user to specify the number of groups to be used in the associated convolution.

    Returns
    -------
    CUDNN_STATUS_SUCCESS
    The group count was set successfully.

    CUDNN_STATUS_BAD_PARAM
    An invalid convolution descriptor was provided
    """
    status = _libcudnn.cudnnSetConvolutionGroupCount(conv_desc, group_count)

    cudnnCheckStatus(status)


_libcudnn.cudnnSetConvolutionMathType.restype = int
_libcudnn.cudnnSetConvolutionMathType.argtypes = [ctypes.c_void_p,
                                                  ctypes.c_int]


def cudnnSetConvolutionMathType(conv_desc, math_type):
    """"
    This function allows the user to specify whether or not the use of tensor op is permitted in the library
    routines associated with a given convolution descriptor.

    Returns
    -------
    CUDNN_STATUS_SUCCESS
    The math type was set successfully.

    CUDNN_STATUS_BAD_PARAM
    Either an invalid convolution descriptor was provided or an invalid math type was specified.
    This function allows the user to specify the number of groups to be used in the associated convolution.
    """
    status = _libcudnn.cudnnSetConvolutionMathType(conv_desc, math_type)

    cudnnCheckStatus(status)


_libcudnn.cudnnGetConvolutionForwardWorkspaceSize.restype = int
_libcudnn.cudnnGetConvolutionForwardWorkspaceSize.argtypes = [ctypes.c_void_p,
                                                              ctypes.c_void_p,
                                                              ctypes.c_void_p,
                                                              ctypes.c_void_p,
                                                              ctypes.c_void_p,
                                                              ctypes.c_int]


def cudnnGetConvolutionForwardWorkspaceSize(handle, src_desc, w_desc,
                                            conv_desc, dest_desc, algo):
    """"
    This function returns the amount of GPU memory workspace the user needs
    to allocate to be able to call cudnnConvolutionForward with the specified algorithm.
    Parameters
    ----------
    handle : cudnnHandle
        Handle to a previously created cuDNN context.
    src_desc : cudnnTensorDescriptor
        Handle to a previously initialized tensor descriptor.
    w_desc : cudnnFilterDescriptor
        Handle to a previously initialized filter descriptor.
    conv_desc : cudnnConvolutionDescriptor
        Previously initialized convolution descriptor.
    dest_desc : cudnnTensorDescriptor
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

    status = _libcudnn.cudnnGetConvolutionForwardWorkspaceSize(handle, src_desc, w_desc,
                                                               conv_desc, dest_desc, algo,
                                                               ctypes.byref(size_in_bytes))
    cudnnCheckStatus(status)

    return size_in_bytes


_libcudnn.cudnnConvolutionForward.restype = int
_libcudnn.cudnnConvolutionForward.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                                              ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                                              ctypes.c_void_p, ctypes.c_int,
                                              ctypes.c_void_p, ctypes.c_size_t,
                                              ctypes.c_void_p, ctypes.c_void_p,
                                              ctypes.c_void_p]


def cudnnConvolutionForward(handle, alpha, src_desc, src_data, w_desc, w,
                            conv_desc, algo, workspace, workspace_size_in_bytes, beta,
                            dest_desc, dest_data):
    """"
    Perform forward convolution. All of the form "output = alpha * Op(inputs) + beta * output".
    This function executes convolutions or cross-correlations over src using the specified
    filters, returning results in dest. Scaling factors alpha and beta can be used to scale
    the input tensor and the output tensor respectively.
    Parameters
    ----------
    handle : cudnnHandle
        Handle to a previously created cuDNN context.
    alpha: float
        Scaling factor with which every element of the input tensor is multiplied.
    src_desc : cudnnTensorDescriptor
        Handle to a previously initialized tensor descriptor.
    src_data : void_p
        Data pointer to GPU memory associated with the tensor descriptor src_desc.
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
    dest_desc : cudnnTensorDescriptor
        Handle to a previously initialized tensor descriptor.
    dest_data : void_p
        Data pointer to GPU memory associated with the tensor descriptor dest_desc.
    """

    data_type = cudnnGetTensor4dDescriptor(dest_desc)[0]
    if data_type == cudnnDataType['CUDNN_DATA_DOUBLE']:
        alpha_ref = ctypes.byref(ctypes.c_double(alpha))
        beta_ref = ctypes.byref(ctypes.c_double(beta))
    else:
        alpha_ref = ctypes.byref(ctypes.c_float(alpha))
        beta_ref = ctypes.byref(ctypes.c_float(beta))

    status = _libcudnn.cudnnConvolutionForward(handle, alpha_ref, src_desc, src_data,
                                               w_desc, w,
                                               conv_desc, algo, workspace,
                                               ctypes.c_size_t(workspace_size_in_bytes),
                                               beta_ref, dest_desc, dest_data)
    cudnnCheckStatus(status)


_libcudnn.cudnnConvolutionBackwardBias.restype = int
_libcudnn.cudnnConvolutionBackwardBias.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                                                   ctypes.c_void_p, ctypes.c_void_p,
                                                   ctypes.c_void_p, ctypes.c_void_p]


def cudnnConvolutionBackwardBias(handle, alpha, src_desc, src_data, beta, dest_desc, dest_data):
    """"
    Compute the gradient wrt the bias.
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
    src_desc : cudnnTensorDescriptor
        Handle to the previously initialized input tensor descriptor.
    src_data : void_p
        Data pointer to GPU memory associated with the tensor descriptor
        src_desc.
    beta: float
        Scaling factor which is applied on every element of the output tensor prior
        to adding the result of the convolution gradient. Note that if beta is zero,
        the output is not read and can contain any uninitialized data (including
        Nan numbers).
    dest_desc : cudnnTensorDescriptor
        Handle to the previously initialized output tensor descriptor.
    dest_data : void_p
        Data pointer to GPU memory associated with the output tensor descriptor
        dest_desc.
    """

    data_type = cudnnGetTensor4dDescriptor(dest_desc)[0]
    if data_type == cudnnDataType['CUDNN_DATA_DOUBLE']:
        alpha_ref = ctypes.byref(ctypes.c_double(alpha))
        beta_ref = ctypes.byref(ctypes.c_double(beta))
    else:
        alpha_ref = ctypes.byref(ctypes.c_float(alpha))
        beta_ref = ctypes.byref(ctypes.c_float(beta))

    status = _libcudnn.cudnnConvolutionBackwardBias(handle, alpha_ref, src_desc, src_data,
                                                    beta_ref, dest_desc, dest_data)
    cudnnCheckStatus(status)


class CudnnConvolutionBwdDataAlgoPerf(ctypes.Structure):
    _fields_ = [("algo", ctypes.c_int),
                ("status", ctypes.c_int),
                ("time", ctypes.c_float),
                ("memory", ctypes.c_size_t)]

    def __str__(self):
        return '(algo=%d, status=%d, time=%f, memory=%d)' % (self.algo,
                                                             self.status,
                                                             self.time,
                                                             self.memory)

    def __repr__(self):
        return self.__str__()


_libcudnn.cudnnFindConvolutionBackwardDataAlgorithm.restype = int
_libcudnn.cudnnFindConvolutionBackwardDataAlgorithm.argtypes = [ctypes.c_void_p,  # handle
                                                                ctypes.c_void_p,  # w_desc
                                                                ctypes.c_void_p,  # dy_desc
                                                                ctypes.c_void_p,  # conv_desc
                                                                ctypes.c_void_p,  # dx_desc
                                                                ctypes.c_int,  # requestAlgoCount
                                                                ctypes.c_void_p,  # returned_algo_count
                                                                ctypes.c_void_p]  # perf_results


def cudnnFindConvolutionBackwardDataAlgorithm(handle, w_desc, dy_desc,
                                              conv_desc, dx_desc,
                                              requested_algo_count):
    perf_results_type = CudnnConvolutionBwdDataAlgoPerf * requested_algo_count
    perf_results = perf_results_type()
    returned_algo_count = ctypes.c_int()
    status = _libcudnn.cudnnFindConvolutionBackwardDataAlgorithm(handle,
                                                                 w_desc,
                                                                 dy_desc,
                                                                 conv_desc,
                                                                 dx_desc,
                                                                 ctypes.c_int(requested_algo_count),
                                                                 ctypes.byref(returned_algo_count),
                                                                 ctypes.cast(perf_results, ctypes.POINTER(
                                                                     CudnnConvolutionBwdDataAlgoPerf)))
    cudnnCheckStatus(status)
    return perf_results[0:returned_algo_count.value]


# _libcudnn.cudnnGetConvolutionBackwardDataAlgorithm.restype = int
# _libcudnn.cudnnGetConvolutionBackwardDataAlgorithm.argtypes = [ctypes.c_void_p,
#                                                                ctypes.c_void_p,
#                                                                ctypes.c_void_p,
#                                                                ctypes.c_void_p,
#                                                                ctypes.c_void_p,
#                                                                ctypes.c_int,
#                                                                ctypes.c_size_t,
#                                                                ctypes.c_void_p]
# def cudnnGetConvolutionBackwardDataAlgorithm(handle, w_desc, dy_desc, conv_desc,
#                                              dx_desc, preference, memoryLimitInbytes):
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
#
_libcudnn.cudnnGetConvolutionBackwardDataWorkspaceSize.restype = int
_libcudnn.cudnnGetConvolutionBackwardDataWorkspaceSize.argtypes = [ctypes.c_void_p,
                                                                   ctypes.c_void_p,
                                                                   ctypes.c_void_p,
                                                                   ctypes.c_void_p,
                                                                   ctypes.c_void_p,
                                                                   ctypes.c_int,  # algo
                                                                   ctypes.c_void_p]


def cudnnGetConvolutionBackwardDataWorkspaceSize(handle, w_desc, dy_desc,
                                                 conv_desc, dx_desc, algo):
    size_in_bytes = ctypes.c_size_t()
    status = _libcudnn.cudnnGetConvolutionBackwardDataWorkspaceSize(handle,
                                                                    w_desc,
                                                                    dy_desc,
                                                                    conv_desc,
                                                                    dx_desc,
                                                                    algo,
                                                                    ctypes.byref(size_in_bytes))
    cudnnCheckStatus(status)
    return size_in_bytes


_libcudnn.cudnnConvolutionBackwardData.restype = int
_libcudnn.cudnnConvolutionBackwardData.argtypes = [ctypes.c_void_p,
                                                   ctypes.c_void_p,
                                                   ctypes.c_void_p, ctypes.c_void_p,
                                                   ctypes.c_void_p, ctypes.c_void_p,
                                                   ctypes.c_void_p,
                                                   ctypes.c_int,
                                                   ctypes.c_void_p, ctypes.c_size_t,
                                                   ctypes.c_void_p,
                                                   ctypes.c_void_p, ctypes.c_void_p]


def cudnnConvolutionBackwardData(handle,
                                 alpha,
                                 w_desc, w,
                                 dy_desc, dy,
                                 conv_desc,
                                 algo,
                                 workspace, workspace_size_in_bytes,
                                 beta,
                                 dx_desc, dx):
    data_type = cudnnGetTensor4dDescriptor(dy_desc)[0]
    if data_type == cudnnDataType['CUDNN_DATA_FLOAT']:
        alpha_ref = ctypes.byref(ctypes.c_float(alpha))
        beta_ref = ctypes.byref(ctypes.c_float(beta))
    else:
        alpha_ref = ctypes.byref(ctypes.c_double(alpha))
        beta_ref = ctypes.byref(ctypes.c_double(beta))

    status = _libcudnn.cudnnConvolutionBackwardData(handle,
                                                    alpha_ref,
                                                    w_desc, w,
                                                    dy_desc, dy,
                                                    conv_desc,
                                                    algo,
                                                    workspace, workspace_size_in_bytes,
                                                    beta_ref,
                                                    dx_desc, dx)
    cudnnCheckStatus(status)


class CudnnConvolutionBwdFilterAlgoPerf(ctypes.Structure):
    _fields_ = [("algo", ctypes.c_int),
                ("status", ctypes.c_int),
                ("time", ctypes.c_float),
                ("memory", ctypes.c_size_t)]

    def __str__(self):
        return '(algo=%d, status=%d, time=%f, memory=%d)' % (self.algo,
                                                             self.status,
                                                             self.time,
                                                             self.memory)

    def __repr__(self):
        return self.__str__()


_libcudnn.cudnnFindConvolutionBackwardFilterAlgorithm.restype = int
_libcudnn.cudnnFindConvolutionBackwardFilterAlgorithm.argtypes = [ctypes.c_void_p,  # handle
                                                                  ctypes.c_void_p,  # x_desc
                                                                  ctypes.c_void_p,  # dy_desc
                                                                  ctypes.c_void_p,  # conv_desc
                                                                  ctypes.c_void_p,  # dw_desc
                                                                  ctypes.c_int,  # requestAlgoCount
                                                                  ctypes.c_void_p,  # returned_algo_count
                                                                  ctypes.c_void_p]  # perf_results


def cudnnFindConvolutionBackwardFilterAlgorithm(handle, x_desc, dy_desc,
                                                conv_desc, dw_desc,
                                                requested_algo_count):
    perf_results_type = CudnnConvolutionBwdFilterAlgoPerf * requested_algo_count
    perf_results = perf_results_type()
    returned_algo_count = ctypes.c_int()
    status = _libcudnn.cudnnFindConvolutionBackwardFilterAlgorithm(handle,
                                                                   x_desc,
                                                                   dy_desc,
                                                                   conv_desc,
                                                                   dw_desc,
                                                                   ctypes.c_int(requested_algo_count),
                                                                   ctypes.byref(returned_algo_count),
                                                                   ctypes.cast(perf_results, ctypes.POINTER(
                                                                       CudnnConvolutionBwdFilterAlgoPerf)))
    cudnnCheckStatus(status)
    return perf_results[0:returned_algo_count.value]


# _libcudnn.cudnnGetConvolutionBackwardFilterAlgorithm.restype = int
# _libcudnn.cudnnGetConvolutionBackwardFilterAlgorithm.argtypes = [ctypes.c_void_p,
#                                                                 ctypes.c_void_p,
#                                                                 ctypes.c_void_p,
#                                                                 ctypes.c_void_p,
#                                                                 ctypes.c_void_p,
#                                                                 ctypes.c_int,
#                                                                 ctypes.c_size_t,
#                                                                 ctypes.c_void_p]
# def cudnnGetConvolutionBackwardFilterAlgorithm(handle, x_desc, dy_desc, conv_desc,
#                                               dw_desc, preference, memoryLimitInbytes):
#    algo = ctypes.c_int()
#    status = _libcudnn.cudnnGetConvolutionBackwardFilterAlgorithm(handle,
#                                                                  x_desc,
#                                                                  dy_desc,
#                                                                  conv_desc,
#                                                                  dw_desc,
#                                                                  preference,
#                                                                  ctypes.c_size_t(memoryLimitInbytes),
#                                                                  ctypes.byref(algo))
#    cudnnCheckStatus(status)
#    return algo
#
#
_libcudnn.cudnnGetConvolutionBackwardFilterWorkspaceSize.restype = int
_libcudnn.cudnnGetConvolutionBackwardFilterWorkspaceSize.argtypes = [ctypes.c_void_p,
                                                                     ctypes.c_void_p,
                                                                     ctypes.c_void_p,
                                                                     ctypes.c_void_p,
                                                                     ctypes.c_void_p,
                                                                     ctypes.c_int,  # algo
                                                                     ctypes.c_void_p]


def cudnnGetConvolutionBackwardFilterWorkspaceSize(handle, x_desc, dy_desc,
                                                   conv_desc, grad_desc, algo):
    size_in_bytes = ctypes.c_size_t()
    status = _libcudnn.cudnnGetConvolutionBackwardFilterWorkspaceSize(handle,
                                                                      x_desc,
                                                                      dy_desc,
                                                                      conv_desc,
                                                                      grad_desc,
                                                                      algo,
                                                                      ctypes.byref(size_in_bytes))
    cudnnCheckStatus(status)
    return size_in_bytes


_libcudnn.cudnnConvolutionBackwardFilter.restype = int
_libcudnn.cudnnConvolutionBackwardFilter.argtypes = [ctypes.c_void_p,
                                                     ctypes.c_void_p,
                                                     ctypes.c_void_p, ctypes.c_void_p,
                                                     ctypes.c_void_p, ctypes.c_void_p,
                                                     ctypes.c_void_p,
                                                     ctypes.c_int,
                                                     ctypes.c_void_p, ctypes.c_size_t,
                                                     ctypes.c_void_p,
                                                     ctypes.c_void_p, ctypes.c_void_p]


def cudnnConvolutionBackwardFilter(handle,
                                   alpha,
                                   x_desc, x,
                                   dy_desc, dy,
                                   conv_desc,
                                   algo,
                                   workspace, workspace_size_in_bytes,
                                   beta,
                                   dw_desc, dw):
    data_type = cudnnGetTensor4dDescriptor(dy_desc)[0]
    if data_type == cudnnDataType['CUDNN_DATA_DOUBLE']:
        alpha_ref = ctypes.byref(ctypes.c_double(alpha))
        beta_ref = ctypes.byref(ctypes.c_double(beta))
    else:
        alpha_ref = ctypes.byref(ctypes.c_float(alpha))
        beta_ref = ctypes.byref(ctypes.c_float(beta))

    status = _libcudnn.cudnnConvolutionBackwardFilter(handle,
                                                      alpha_ref,
                                                      x_desc, x,
                                                      dy_desc, dy,
                                                      conv_desc,
                                                      algo,
                                                      workspace, workspace_size_in_bytes,
                                                      beta_ref,
                                                      dw_desc, dw)
    cudnnCheckStatus(status)


_libcudnn.cudnnSoftmaxForward.restype = int
_libcudnn.cudnnSoftmaxForward.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_void_p,
                                          ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                                          ctypes.c_void_p, ctypes.c_void_p]


def cudnnSoftmaxForward(handle, algorithm, mode, alpha, src_desc, src_data, beta, dest_desc, dest_data):
    """"
    This routing computes the softmax function
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
    src_desc : cudnnTensorDescriptor
        Handle to the previously initialized input tensor descriptor.
    src_data : void_p
        Data pointer to GPU memory associated with the tensor descriptor
        src_desc.
    beta: float
        Scaling factor which is applied on every element of the output tensor prior
        to adding the result of the activation Note that if beta is zero, the output
        is not read and can contain any uninitialized data (including Nan numbers).
    dest_desc : cudnnTensorDescriptor
        Handle to the previously initialized output tensor descriptor.
    dest_data : void_p
        Data pointer to GPU memory associated with the output tensor descriptor
        dest_desc.
    """

    data_type = cudnnGetTensor4dDescriptor(dest_desc)[0]
    if data_type == cudnnDataType['CUDNN_DATA_DOUBLE']:
        alpha_ref = ctypes.byref(ctypes.c_double(alpha))
        beta_ref = ctypes.byref(ctypes.c_double(beta))
    else:
        alpha_ref = ctypes.byref(ctypes.c_float(alpha))
        beta_ref = ctypes.byref(ctypes.c_float(beta))

    status = _libcudnn.cudnnSoftmaxForward(handle, algorithm, mode, alpha_ref,
                                           src_desc, src_data, beta_ref,
                                           dest_desc, dest_data)
    cudnnCheckStatus(status)


_libcudnn.cudnnSoftmaxBackward.restype = int
_libcudnn.cudnnSoftmaxBackward.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int,
                                           ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                                           ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                                           ctypes.c_void_p, ctypes.c_void_p]


def cudnnSoftmaxBackward(handle, algorithm, mode, alpha, src_desc, src_data, src_diff_esc,
                         src_diff_data, beta, dest_diff_desc, dest_diff_data):
    """"
    This routine computes the gradient of the softmax function.
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
    src_desc : cudnnTensorDescriptor
        Handle to the previously initialized input tensor descriptor.
    src_data : void_p
        Data pointer to GPU memory associated with the tensor descriptor
        src_desc.
    src_diff_esc : cudnnTensorDescriptor
        Handle to the previously initialized input differential tensor descriptor.
    src_diff_data : void_p
        Data pointer to GPU memory associated with the tensor descriptor
        src_diff_data.
    beta: float
        Scaling factor which is applied on every element of the output tensor prior
        to adding the result of the activation Note that if beta is zero, the output
        is not read and can contain any uninitialized data (including Nan numbers).
    dest_diff_desc : cudnnTensorDescriptor
        Handle to the previously initialized output differential tensor descriptor.
    dest_diff_data : void_p
        Data pointer to GPU memory associated with the output tensor descriptor
        dest_diff_desc.
    """

    data_type = cudnnGetTensor4dDescriptor(dest_diff_desc)[0]
    if data_type == cudnnDataType['CUDNN_DATA_DOUBLE']:
        alpha_ref = ctypes.byref(ctypes.c_double(alpha))
        beta_ref = ctypes.byref(ctypes.c_double(beta))
    else:
        alpha_ref = ctypes.byref(ctypes.c_float(alpha))
        beta_ref = ctypes.byref(ctypes.c_float(beta))

    status = _libcudnn.cudnnSoftmaxBackward(handle, algorithm, mode, alpha_ref,
                                            src_desc, src_data,
                                            src_diff_esc, src_diff_data, beta_ref,
                                            dest_diff_desc, dest_diff_data)
    cudnnCheckStatus(status)


_libcudnn.cudnnCreateDropoutDescriptor.restype = int
_libcudnn.cudnnCreateDropoutDescriptor.argtypes = [ctypes.c_void_p]


def cudnnCreateDropoutDescriptor():
    """"
    Create dropout descriptor.
    This function creates a dropout descriptor object by allocating the memory needed to
    hold its opaque structure,
    Returns
    -------
    dropout_esc : cudnnDropoutDescriptor
        Newly allocated dropout descriptor.
    """

    dropout_esc = ctypes.c_void_p()
    status = _libcudnn.cudnnCreateDropoutDescriptor(ctypes.byref(dropout_esc))
    cudnnCheckStatus(status)

    return dropout_esc.value


_libcudnn.cudnnSetDropoutDescriptor.restype = int
_libcudnn.cudnnSetDropoutDescriptor.argtypes = [ctypes.c_void_p, ctypes.c_void_p,
                                                ctypes.c_float, ctypes.c_void_p,
                                                ctypes.c_size_t, ctypes.c_ulonglong]


def cudnnSetDropoutDescriptor(drop_desc, handle, dropout, states, state_size_in_bytes, seed):
    status = _libcudnn.cudnnSetDropoutDescriptor(drop_desc, handle, dropout,
                                                 states, state_size_in_bytes, seed)
    cudnnCheckStatus(status)


_libcudnn.cudnnDropoutGetReserveSpaceSize.restype = int
_libcudnn.cudnnDropoutGetReserveSpaceSize.argtypes = [ctypes.c_void_p]


def cudnnDropoutGetReserveSpaceSize(x_desc):
    """"
    This function is used to query the amount of reserve needed to run dropout
    with the input dimensions given by x_desc
    Returns
    -------
    The size in bytes
    """

    size_in_bytes = ctypes.c_size_t()

    status = _libcudnn.cudnnDropoutGetReserveSpaceSize(x_desc, ctypes.byref(size_in_bytes))
    cudnnCheckStatus(status)

    return size_in_bytes


_libcudnn.cudnnDropoutGetStatesSize.restype = int
_libcudnn.cudnnDropoutGetStatesSize.argtypes = [ctypes.c_void_p]


def cudnnDropoutGetStatesSize(handle):
    """"
    This function is used to query the amount of space required to store
    the states of the random number generators used by cudnnDropoutForward() function
    Returns
    -------
    The size in bytes
    """

    size_in_bytes = ctypes.c_size_t()

    status = _libcudnn.cudnnDropoutGetStatesSize(handle, ctypes.byref(size_in_bytes))
    cudnnCheckStatus(status)

    return size_in_bytes


_libcudnn.cudnnDropoutForward.restype = int
_libcudnn.cudnnDropoutForward.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                                          ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                                          ctypes.c_void_p, ctypes.c_size_t]


def cudnnDropoutForward(handle, dropout_esc, x_desc, x, y_desc, y, reserve_space, reserve_space_size_in_bytes):
    status = _libcudnn.cudnnDropoutForward(handle, dropout_esc, x_desc, x, y_desc, y,
                                           reserve_space, ctypes.c_size_t(reserve_space_size_in_bytes))
    cudnnCheckStatus(status)


_libcudnn.cudnnDropoutBackward.restype = int
_libcudnn.cudnnDropoutBackward.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                                           ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                                           ctypes.c_void_p, ctypes.c_void_p]


def cudnnDropoutBackward(handle, dropout_esc, dy_desc, dy, dx_desc, dx, reserve_space, reserve_space_size_in_bytes):
    status = _libcudnn.cudnnDropoutBackward(handle, dropout_esc, dy_desc, dy, dx_desc, dx,
                                            reserve_space, reserve_space_size_in_bytes)

    cudnnCheckStatus(status)


_libcudnn.cudnnCreatePoolingDescriptor.restype = int
_libcudnn.cudnnCreatePoolingDescriptor.argtypes = [ctypes.c_void_p]


def cudnnCreatePoolingDescriptor():
    """"
    Create pooling descriptor.
    This function creates a pooling descriptor object by allocating the memory needed to
    hold its opaque structure,
    Returns
    -------
    pooling_desc : cudnnPoolingDescriptor
        Newly allocated pooling descriptor.
    """

    pooling_desc = ctypes.c_void_p()
    status = _libcudnn.cudnnCreatePoolingDescriptor(ctypes.byref(pooling_desc))
    cudnnCheckStatus(status)

    return pooling_desc.value


_libcudnn.cudnnSetPooling2dDescriptor.restype = int
_libcudnn.cudnnSetPooling2dDescriptor.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int,
                                                  ctypes.c_int, ctypes.c_int,
                                                  ctypes.c_int, ctypes.c_int,
                                                  ctypes.c_int, ctypes.c_int]


def cudnnSetPooling2dDescriptor(pooling_desc, mode, nan, window_height, window_width,
                                vertical_padding, horizontal_padding, vertical_stride, horizontal_stride):
    """"
    Initialize a 2D pooling descriptor.
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

    status = _libcudnn.cudnnSetPooling2dDescriptor(pooling_desc, mode, nan, window_height,
                                                   window_width, vertical_padding, horizontal_padding,
                                                   vertical_stride, horizontal_stride)
    cudnnCheckStatus(status)


_libcudnn.cudnnGetPooling2dDescriptor.restype = int
_libcudnn.cudnnGetPooling2dDescriptor.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                                                  ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                                                  ctypes.c_void_p, ctypes.c_void_p]


def cudnnGetPooling2dDescriptor(pooling_desc):
    """"
    This function queries a previously created pooling descriptor object.
    Parameters
    ----------
    pooling_desc : cudnnPoolingDescriptor
    Handle to a previously created 2D pooling descriptor.
    Returns
    -------
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

    mode = ctypes.c_int()
    window_height = ctypes.c_int()
    window_width = ctypes.c_int()
    vertical_padding = ctypes.c_int()
    horizontal_padding = ctypes.c_int()
    vertical_stride = ctypes.c_int()
    horizontal_stride = ctypes.c_int()

    status = _libcudnn.cudnnGetPooling2dDescriptor(pooling_desc, ctypes.byref(mode), ctypes.byref(window_height),
                                                   ctypes.byref(window_width), ctypes.byref(vertical_padding),
                                                   ctypes.byref(horizontal_padding), ctypes.byref(vertical_stride),
                                                   ctypes.byref(horizontal_stride))
    cudnnCheckStatus(status)

    return mode.value, window_height.value, window_width.value, vertical_stride.value, horizontal_stride.value


_libcudnn.cudnnDestroyPoolingDescriptor.restype = int
_libcudnn.cudnnDestroyPoolingDescriptor.argtypes = [ctypes.c_void_p]


def cudnnDestroyPoolingDescriptor(pooling_desc):
    """"
    This function destroys a previously created pooling descriptor object.
    Parameters
    ----------
    pooling_desc : cudnnPoolingDescriptor
    """

    status = _libcudnn.cudnnDestroyPoolingDescriptor(pooling_desc)
    cudnnCheckStatus(status)


_libcudnn.cudnnGetPooling2dForwardOutputDim.restype = int
_libcudnn.cudnnGetPooling2dForwardOutputDim.argtypes = [ctypes.c_void_p, ctypes.c_void_p,
                                                        ctypes.c_void_p, ctypes.c_void_p,
                                                        ctypes.c_void_p, ctypes.c_void_p]


def cudnnGetPooling2dForwardOutputDim(pooling_desc, input_desc):
    """"
    This function provides the output dimensions of a tensor after 2d pooling has been applied.

    Each dimension h and w of the output images is computed as follows:
        outputDim = 1 + (inputDim + 2*padding - windowDim)/poolingStride;

    Parameters
    ----------
    pooling_desc : Input
        Handle to a previously initialized pooling descriptor.

    input_desc : Input
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

    status = _libcudnn.cudnnGetPooling2dForwardOutputDim(pooling_desc, input_desc, ctypes.byref(n),
                                                         ctypes.byref(c), ctypes.byref(h), ctypes.byref(w))
    cudnnCheckStatus(status)

    return n.value, c.value, h.value, w.value


_libcudnn.cudnnPoolingForward.restype = int
_libcudnn.cudnnPoolingForward.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                                          ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                                          ctypes.c_void_p, ctypes.c_void_p]


def cudnnPoolingForward(handle, pooling_desc, alpha, src_desc, src_data, beta, dest_desc, dest_data):
    """"
    Perform pooling.
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
    src_desc : cudnnTensorDescriptor
        Handle to the previously initialized input tensor descriptor.
    src_data : void_p
        Data pointer to GPU memory associated with the tensor descriptor
        src_desc.
    beta: float
        Scaling factor which is applied on every element of the output tensor prior
        to adding the result of the activation Note that if beta is zero, the output
        is not read and can contain any uninitialized data (including Nan numbers).
    dest_desc : cudnnTensorDescriptor
        Handle to the previously initialized output tensor descriptor.
    dest_data : void_p
        Data pointer to GPU memory associated with the output tensor descriptor
        dest_desc.
    """

    data_type = cudnnGetTensor4dDescriptor(dest_desc)[0]
    if data_type == cudnnDataType['CUDNN_DATA_DOUBLE']:
        alpha_ref = ctypes.byref(ctypes.c_double(alpha))
        beta_ref = ctypes.byref(ctypes.c_double(beta))
    else:
        alpha_ref = ctypes.byref(ctypes.c_float(alpha))
        beta_ref = ctypes.byref(ctypes.c_float(beta))

    status = _libcudnn.cudnnPoolingForward(handle, pooling_desc, alpha_ref,
                                           src_desc, src_data, beta_ref,
                                           dest_desc, dest_data)
    cudnnCheckStatus(status)


_libcudnn.cudnnPoolingBackward.restype = int
_libcudnn.cudnnPoolingBackward.argtypes = [ctypes.c_void_p, ctypes.c_void_p,
                                           ctypes.c_void_p, ctypes.c_void_p,
                                           ctypes.c_void_p, ctypes.c_void_p,
                                           ctypes.c_void_p, ctypes.c_void_p,
                                           ctypes.c_void_p, ctypes.c_void_p,
                                           ctypes.c_void_p, ctypes.c_void_p]


def cudnnPoolingBackward(handle, pooling_desc, alpha, src_desc, src_data, src_diff_esc,
                         src_diff_data, dest_desc, dest_data, beta, dest_diff_desc, dest_diff_data):
    """"
    Gradients wrt the pooling operation.
    This function computes the gradient of a pooling operation.
    Parameters
    ----------
    handle : cudnnHandle
        Handle to a previously created cuDNN context.
    pooling_desc : cudnnPoolingDescriptor
        Handle to the previously initialized pooling descriptor.
    alpha: float
        Scaling factor with which every element of the input tensors is multiplied.
    src_desc : cudnnTensorDescriptor
        Handle to the previously initialized input tensor descriptor.
    src_data : void_p
        Data pointer to GPU memory associated with the tensor descriptor
        src_desc.
    src_diff_esc : cudnnTensorDescriptor
        Handle to the previously initialized input differential tensor descriptor.
    src_diff_data : void_p
        Data pointer to GPU memory associated with the tensor descriptor
        src_diff_data.
    dest_desc : cudnnTensorDescriptor
        Handle to the previously initialized output tensor descriptor.
    dest_data : void_p
        Data pointer to GPU memory associated with the output tensor descriptor
        dest_desc.
    beta: float
        Scaling factor which is applied on every element of the output tensor prior
        to adding the result of the activation Note that if beta is zero, the output
        is not read and can contain any uninitialized data (including Nan numbers).
    dest_diff_desc : cudnnTensorDescriptor
        Handle to the previously initialized output differential tensor descriptor.
    dest_diff_data : void_p
        Data pointer to GPU memory associated with the output tensor descriptor
        dest_diff_desc.
    """

    data_type = cudnnGetTensor4dDescriptor(dest_desc)[0]
    if data_type == cudnnDataType['CUDNN_DATA_DOUBLE']:
        alpha_ref = ctypes.byref(ctypes.c_double(alpha))
        beta_ref = ctypes.byref(ctypes.c_double(beta))
    else:
        alpha_ref = ctypes.byref(ctypes.c_float(alpha))
        beta_ref = ctypes.byref(ctypes.c_float(beta))

    status = _libcudnn.cudnnPoolingBackward(handle, pooling_desc, alpha_ref,
                                            src_desc, src_data, src_diff_esc, src_diff_data,
                                            dest_desc, dest_data, beta_ref,
                                            dest_diff_desc, dest_diff_data)
    cudnnCheckStatus(status)


_libcudnn.cudnnDeriveBNTensorDescriptor.restype = int
_libcudnn.cudnnDeriveBNTensorDescriptor.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int]


def cudnnDeriveBNTensorDescriptor(derive_bn_desc, x_desc, mode):
    """
    This function derives a secondary tensor descriptor for the batch normalization
    scale, invVariance, bn_bias, and bn_scale subtensors from the layer's x data descriptor.

    derivedBnDesc
    Output. Handle to a previously created tensor descriptor.
    x_desc
    Input. Handle to a previously created and initialized layer's x data descriptor.
    mode
    Input. Batch normalization layer mode of operation.

    """
    status = _libcudnn.cudnnDeriveBNTensorDescriptor(derive_bn_desc, x_desc, mode)

    cudnnCheckStatus(status)


_libcudnn.cudnnBatchNormalizationBackward.restype = int
_libcudnn.cudnnBatchNormalizationBackward.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p,
                                                      ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                                                      ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                                                      ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                                                      ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                                                      ctypes.c_void_p, ctypes.c_double, ctypes.c_void_p,
                                                      ctypes.c_void_p]

_libcudnn.cudnnCreateSeqDataDescriptor.restype = int
_libcudnn.cudnnCreateSeqDataDescriptor.argtypes = [ctypes.c_void_p]


def cudnnCreateSeqDataDescriptor():
    """
    Create a SeqData descriptor object.
    Allocates a cudnnSeqDataDescriptor_t structure and returns a pointer to it.
    Returns
    -------
    seqdata_descriptor : int
        SeqData descriptor.
    """

    seqdata = ctypes.c_void_p()
    status = _libcudnn.cudnnCreateSeqDataDescriptor(ctypes.byref(seqdata))
    cudnnCheckStatus(status)
    return seqdata.value


_libcudnn.cudnnSetSeqDataDescriptor.restype = int
_libcudnn.cudnnSetSeqDataDescriptor.argtypes = [ctypes.c_void_p,  # seqDataDesc
                                                ctypes.c_int,  # dataType
                                                ctypes.c_int,  # nbDims
                                                ctypes.c_void_p,  # dimA[]
                                                ctypes.c_void_p,  # axes[]
                                                ctypes.c_size_t,  # seqLengthArraySize
                                                ctypes.c_void_p,  # seqLengthArray
                                                ctypes.c_void_p]  # paddingFill


def cudnnSetSeqDataDescriptor(seqDataDesc, dataType, nbDims, dimA, axes, seqLengthArraySize, seqLengthArray, paddingFill):
    """
    Initialize a previously created SeqData object.
    This function initializes a previously created sequence data descriptor object. In the most 
    simplified view, this descriptor defines dimensions (dimA) and the data layout (axes) of a 
    four-dimensional tensor. All four dimensions of the sequence data descriptor have unique 
    identifiers that can be used to index the dimA[] array.
    Parameters
    ----------
    seqDataDesc : cudnnSeqDataDescriptor
        Pointer to a previously created sequence data descriptor.
    dataType : cudnnDataType
        Data type of the sequence data buffer (CUDNN_DATA_HALF, CUDNN_DATA_FLOAT or 
        CUDNN_DATA_DOUBLE).
    nbDims : int
        Must be 4. The number of active dimensions in dimA[] and axes[] arrays. Both arrays should 
        be declared to contain at least CUDNN_SEQDATA_DIM_COUNT elements.
    dimA : int[]
        Integer array specifying sequence data dimensions. Use the cudnnSeqDataAxis_t enumerated 
        type to index all active dimA[] elements.
    axes : cudnnSeqDataAxis[]
        Array of cudnnSeqDataAxis_t that defines the layout of sequence data in memory. The first 
        nbDims elements of axes[] should be initialized with the outermost dimension in axes[0] and 
        the innermost dimension in axes[nbDims-1].
    seqLengthArraySize : int
        Number of elements in the sequence length array, seqLengthArray[].
    seqLengthArray : int
        An integer array that defines all sequence lengths of the container.
    paddingFill : void
        Must be NULL. Pointer to a value of dataType that is used to fill up output vectors beyond 
        the valid length of each sequence or NULL to ignore this setting.
    """
    dimARef = dimA.ctypes.data_as(ctypes.POINTER(ctypes.c_int32))
    axesRef = axes.ctypes.data_as(ctypes.POINTER(ctypes.c_int32))
    seqLengthArrayRef = seqLengthArray.ctypes.data_as(ctypes.POINTER(ctypes.c_int32))
    status = _libcudnn.cudnnSetSeqDataDescriptor(seqDataDesc, dataType, nbDims, dimARef, axesRef,
                                                 seqLengthArraySize, seqLengthArrayRef, paddingFill)
    cudnnCheckStatus(status)


_libcudnn.cudnnDestroySeqDataDescriptor.restype = int
_libcudnn.cudnnDestroySeqDataDescriptor.argtypes = [ctypes.c_void_p]


def cudnnDestroySeqDataDescriptor(seqDataDesc):
    """"
    Destroy a SeqData descriptor.
    This function destroys a previously created SeqData descriptor object.
    Parameters
    ----------
    seqDataDesc : cudnnSeqDataDescriptor
        Previously allocated SeqData descriptor object.
    """

    status = _libcudnn.cudnnDestroySeqDataDescriptor(seqDataDesc)
    cudnnCheckStatus(status)


_libcudnn.cudnnCreateAttnDescriptor.restype = int
_libcudnn.cudnnCreateAttnDescriptor.argtypes = [ctypes.c_void_p]


def cudnnCreateAttnDescriptor():
    """
    Create a attnDesc descriptor object.
    Allocates a cudnnAttnDescriptor_t structure and returns a pointer to it.
    Returns
    -------
    cudnnAttnDescriptor : int
        attnDesc descriptor.
    """

    attnDesc = ctypes.c_void_p()
    status = _libcudnn.cudnnCreateAttnDescriptor(ctypes.byref(attnDesc))
    cudnnCheckStatus(status)
    return attnDesc.value


_libcudnn.cudnnSetAttnDescriptor.restype = int
_libcudnn.cudnnSetAttnDescriptor.argtypes = [ctypes.c_void_p,  # attnDesc
                                             ctypes.c_void_p,  # attnMode
                                             ctypes.c_int,  # nHeads
                                             ctypes.c_double,  # smScaler
                                             ctypes.c_void_p,  # dataType
                                             ctypes.c_void_p,  # computePrec
                                             ctypes.c_void_p,  # mathType
                                             ctypes.c_void_p, ctypes.c_void_p,  # attnDropoutDesc, postDropoutDesc
                                             ctypes.c_int, ctypes.c_int, ctypes.c_int,  # qSize, kSize, vSize
                                             ctypes.c_int, ctypes.c_int, ctypes.c_int,  # qProjSize, kProjSize, vProjSize
                                             ctypes.c_int, ctypes.c_int,  # qoMaxSeqLength, kvMaxSeqLength
                                             ctypes.c_int, ctypes.c_int]  # maxBatchSize, maxBeamSize


def cudnnSetAttnDescriptor(attnDesc, attnMode, nHeads, smScaler, dataType, computePrec, mathType,
                           attnDropoutDesc, postDropoutDesc, qSize, kSize, vSize, qProjSize, kProjSize,
                           vProjSize, oProjSize, qoMaxSeqLength, kvMaxSeqLength, maxBatchSize, maxBeamSize):
    """
    This function configures a multi-head attention descriptor that was previously created using 
    the cudnnCreateAttnDescriptor() function. The function sets attention parameters that are 
    necessary to compute internal buffer sizes, dimensions of weight and bias tensors, or to 
    select optimized code paths.
    Parameters
    ----------
    attnDesc : cudnnAttnDescriptor_t
        Output. Attention descriptor to be configured.
    attnMode : unsigned
        Input. Enables various attention options that do not require additional numerical values. 
        The user should assign a preferred set of bitwise OR-ed flags to this argument.
    nHeads : int
        Input. Number of attention heads.
    smScaler : double
        Input. Softmax smoothing (1.0 >= smScaler >= 0.0) or sharpening (smScaler > 1.0) coefficient. 
        Negative values are not accepted.
    dataType : cudnnDataType_t
        Input. Data type used to represent attention inputs, attention weights and attention outputs.
    computePrec : cudnnDataType_t
        Input. Compute precision.
    mathType : cudnnMathType_t
        Input. NVIDIA Tensor Core settings.
    attnDropoutDesc : cudnnDropoutDescriptor_t
        Input. Descriptor of the dropout operation applied to the softmax output. See the table below 
        for a list of unsupported features.
    postDropoutDesc : cudnnDropoutDescriptor_t
        Input. Descriptor of the dropout operation applied to the multi-head attention output, just 
        before the point where residual connections are added.
    qSize, kSize, vSize : int
        Input. Q , K , V embedding vector lengths.
    qProjSize, kProjSize, vProjSize : int
        Input. Q , K , V embedding vector lengths after input projections. Use zero to disable the 
        corresponding projection.
    oProjSize : int
        Input. The h i vector length after the output projection. Use zero to disable this projection.
    qoMaxSeqLength : int
        Input. Largest sequence length expected in sequence data descriptors related to Q , O , dQ 
        and dO inputs and outputs.
    kvMaxSeqLength : int
        Input. Largest sequence length expected in sequence data descriptors related to K , V , dK 
        and dV inputs and outputs.
    maxBatchSize : int
        Input. Largest batch size expected in any cudnnSeqDataDescriptor_t container.
    maxBeamSize : int
        Input. Largest beam size expected in any cudnnSeqDataDescriptor_t container.
    """
    status = _libcudnn.cudnnSetAttnDescriptor(attnDesc, attnMode, nHeads, smScaler, dataType, computePrec,
                                              mathType, attnDropoutDesc, postDropoutDesc, qSize, kSize, vSize,
                                              qProjSize, kProjSize, vProjSize, oProjSize, qoMaxSeqLength,
                                              kvMaxSeqLength, maxBatchSize, maxBeamSize)
    cudnnCheckStatus(status)


_libcudnn.cudnnDestroyAttnDescriptor.restype = int
_libcudnn.cudnnDestroyAttnDescriptor.argtypes = [ctypes.c_void_p]


def cudnnDestroyAttnDescriptor(attnDesc):
    """"
    Destroy a Attn descriptor.
    This function destroys a previously created Attn descriptor object.
    Parameters
    ----------
    attnDesc : cudnnAttnDescriptor
        Previously allocated Attn descriptor object.
    """

    status = _libcudnn.cudnnDestroyAttnDescriptor(attnDesc)
    cudnnCheckStatus(status)


_libcudnn.cudnnGetMultiHeadAttnWeights.restype = int
_libcudnn.cudnnGetMultiHeadAttnWeights.argtypes = [ctypes.c_void_p, ctypes.c_void_p,
                                                   ctypes.c_void_p, ctypes.c_size_t,
                                                   ctypes.c_void_p, ctypes.c_void_p,
                                                   ctypes.c_void_p]


def cudnnGetMultiHeadAttnWeights(handle, attnDesc, wKind, weightSizeInBytes, weights):
    """"
    This function obtains the shape of the weight or bias tensor. It also retrieves the start address 
    of tensor data located in the weight buffer. Use the wKind argument to select a particular tensor. 
    For more information, see cudnnMultiHeadAttnWeightKind_t for the description of the enumerant type.

    Parameters
    ----------
    handle : cudnnHandle_t
        The current cuDNN context handle.
    attnDesc : cudnnAttnDescriptor_t
        A previously configured attention descriptor.
    wKind : cudnnMultiHeadAttnWeightKind_t
        Enumerant type to specify which weight or bias tensor should be retrieved.
    weightSizeInBytes : size_t
        Buffer size that stores all multi-head attention weights and biases.
    weights : void
    Input. Pointer to the weight buffer in the host or device memory.
    Returns
    -------
    wDesc : cudnnTensorDescriptor_t
        The descriptor specifying weight or bias tensor shape. For weights, the wDesc.dimA[] array has 
        three elements: [nHeads, projected size, original size]. For biases, the wDesc.dimA[] array also 
        has three elements: [nHeads, projected size, 1]. The wDesc.strideA[] array describes how tensor 
        elements are arranged in memory.
    wAddr : void
        Pointer to a location where the start address of the requested tensor should be written. When the 
        corresponding projection is disabled, the address written to wAddr is NULL.
    """
    wDesc = ctypes.c_void_p()
    status = _libcudnn.cudnnCreateTensorDescriptor(ctypes.byref(wDesc))
    wAddr = (ctypes.POINTER(ctypes.c_void_p) * 1)()

    status = _libcudnn.cudnnGetMultiHeadAttnWeights(handle, attnDesc, wKind, weightSizeInBytes, weights,
                                                    wDesc, ctypes.byref(wAddr))
    cudnnCheckStatus(status)
    wAddr = wAddr[0]
    return wDesc, wAddr


_libcudnn.cudnnGetMultiHeadAttnBuffers.restype = int
_libcudnn.cudnnGetMultiHeadAttnBuffers.argtypes = [ctypes.c_void_p,  # handle
                                                   ctypes.c_void_p]  # attnDesc


def cudnnGetMultiHeadAttnBuffers(handle, attnDesc):
    """"
    This function computes weight, work, and reserve space buffer sizes used by the following functions:
        cudnnMultiHeadAttnForward()
        cudnnMultiHeadAttnBackwardData()
        cudnnMultiHeadAttnBackwardWeights()
    Returns
    -------
    weightSizeInBytes : size_t
        The size in bytes
    workSpaceSizeInBytes : size_t
        The size in bytes
    reserveSpaceSizeInBytes : size_t
        The size in bytes
    """

    weightSizeInBytes = ctypes.c_size_t()
    workSpaceSizeInBytes = ctypes.c_size_t()
    reserveSpaceSizeInBytes = ctypes.c_size_t()

    status = _libcudnn.cudnnGetMultiHeadAttnBuffers(handle, attnDesc, ctypes.byref(weightSizeInBytes),
                                                    ctypes.byref(workSpaceSizeInBytes), ctypes.byref(reserveSpaceSizeInBytes))
    cudnnCheckStatus(status)

    return weightSizeInBytes, workSpaceSizeInBytes, reserveSpaceSizeInBytes


_libcudnn.cudnnMultiHeadAttnForward.restype = int
_libcudnn.cudnnMultiHeadAttnForward.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int,
                                                ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                                                ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                                                ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                                                ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                                                ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p,
                                                ctypes.c_size_t, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p]


def cudnnMultiHeadAttnForward(handle, attnDesc, currIdx, loWinIdx, hiWinIdx, devSeqLengthsQO, devSeqLengthsKV,
                              qDesc, queries, residuals, kDesc, keys, vDesc, values, oDesc, out,
                              weightSizeInBytes, weights, workSpaceSizeInBytes, workSpace, reserveSpaceSizeInBytes,
                              reserveSpace):
    """"
    The cudnnMultiHeadAttnForward() function computes the forward responses of the multi-head attention layer. 
    When reserveSpaceSizeInBytes=0 and reserveSpace=NULL, the function operates in the inference mode in which 
    backward (gradient) functions are not invoked, otherwise, the training mode is assumed. In the training mode, 
    the reserve space is used to pass intermediate results from cudnnMultiHeadAttnForward() to 
    cudnnMultiHeadAttnBackwardData() and from cudnnMultiHeadAttnBackwardData() to cudnnMultiHeadAttnBackwardWeights().
    Parameters
    ----------
    handle
        Input. The current cuDNN context handle.
    attnDesc
        Input. A previously initialized attention descriptor.
    currIdx
        Input. Time-step in queries to process. When the currIdx argument is negative, all Q time-steps are processed. 
        When currIdx is zero or positive, the forward response is computed for the selected time-step only. The latter 
        input can be used in inference mode only, to process one time-step while updating the next attention window and 
        Q, R, K, V inputs in-between calls.
    loWinIdx[], hiWinIdx[]
        Input. Two host integer arrays specifying the start and end indices of the attention window for each Q time-step. 
        The start index in K, V sets is inclusive, and the end index is exclusive.
    devSeqLengthsQO[]
        Input. Device array specifying sequence lengths of query, residual, and output sequence data.
    devSeqLengthsKV[]
        Input. Device array specifying sequence lengths of key and value input data.
    qDesc
        Input. Descriptor for the query and residual sequence data.
    queries
        Input. Pointer to queries data in the device memory.
    residuals
        Input. Pointer to residual data in device memory. Set this argument to NULL if no residual connections are 
        required.
    kDesc
        Input. Descriptor for the keys sequence data.
    keys
        Input. Pointer to keys data in device memory.
    vDesc
        Input. Descriptor for the values sequence data.
    values
        Input. Pointer to values data in device memory.
    oDesc
        Input. Descriptor for the multi-head attention output sequence data.
    out
        Output. Pointer to device memory where the output response should be written.
    weightSizeInBytes
        Input. Size of the weight buffer in bytes where all multi-head attention trainable parameters are stored.
    weights
        Input. Pointer to the weight buffer in device memory.
    workSpaceSizeInBytes
        Input. Size of the work-space buffer in bytes used for temporary API storage.
    workSpace
        Input/Output. Pointer to the work-space buffer in device memory.
    reserveSpaceSizeInBytes
        Input. Size of the reserve-space buffer in bytes used for data exchange between forward and backward (gradient) 
        API calls. This parameter should be zero in the inference mode and non-zero in the training mode.
    reserveSpace
        Input/Output. Pointer to the reserve-space buffer in device memory. This argument should be NULL in inference mode 
        and non-NULL in the training mode.
    """
    status = _libcudnn.cudnnMultiHeadAttnForward(handle, attnDesc, currIdx, loWinIdx.ctypes.data, hiWinIdx.ctypes.data,
                                                 devSeqLengthsQO, devSeqLengthsKV,
                                                 qDesc, queries, residuals, kDesc, keys, vDesc, values, oDesc, out,
                                                 weightSizeInBytes, weights, workSpaceSizeInBytes, workSpace, reserveSpaceSizeInBytes,
                                                 reserveSpace)
    cudnnCheckStatus(status)


_libcudnn.cudnnMultiHeadAttnBackwardData.restype = int
_libcudnn.cudnnMultiHeadAttnBackwardData.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                                                     ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                                                     ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                                                     ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                                                     ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                                                     ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t,
                                                     ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p,
                                                     ctypes.c_size_t, ctypes.c_void_p]


def cudnnMultiHeadAttnBackwardData(handle, attnDesc, loWinIdx, hiWinIdx, devSeqLengthsDQDO, devSeqLengthsDKDV,
                                   doDesc, dout, dqDesc, dqueries, queries, dkDesc, dkeys, keys, dvDesc, dvalues, values,
                                   weightSizeInBytes, weights, workSpaceSizeInBytes, workSpace, reserveSpaceSizeInBytes,
                                   reserveSpace):
    """"
    This function computes exact, first-order derivatives of the multi-head attention block with respect to its inputs: 
    Q, K, V. If y=F(x) is a vector-valued function that represents the multi-head attention layer and it takes some vector 
    x ϵ ℝ n as an input (with all other parameters and inputs constant), and outputs vector y ϵ ℝ m , then 
    cudnnMultiHeadAttnBackwardData() computes the result of ∂ y i / ∂ x j T δ out where δ out is the m × 1 gradient of the 
    loss function with respect to multi-head attention outputs. The δ out gradient is back propagated through prior layers 
    of the deep learning model. ∂ y i / ∂ x j is the m × n Jacobian matrix of F(x). The input is supplied via the dout argument 
    and gradient results for Q, K, V are written to the dqueries, dkeys, and dvalues buffers.
    Parameters
    ----------
    handle
        Input. The current cuDNN context handle.
    attnDesc
        Input. A previously initialized attention descriptor.
    currIdx
        Input. Time-step in queries to process. When the currIdx argument is negative, all Q time-steps are processed. 
        When currIdx is zero or positive, the forward response is computed for the selected time-step only. The latter 
        input can be used in inference mode only, to process one time-step while updating the next attention window and 
        Q, R, K, V inputs in-between calls.
    loWinIdx[], hiWinIdx[]
        Input. Two host integer arrays specifying the start and end indices of the attention window for each Q time-step. 
        The start index in K, V sets is inclusive, and the end index is exclusive.
    devSeqLengthsDQDO[]
        Input. Device array containing a copy of the sequence length array from the dqDesc or doDesc sequence data descriptor.
    devSeqLengthsDKDV[]
        Input. Device array containing a copy of the sequence length array from the dkDesc or dvDesc sequence data descriptor.
    doDesc
        Input. Descriptor for the δ out gradients (vectors of partial derivatives of the loss function with respect to the multi-head attention outputs).
    dout
        Pointer to δ out gradient data in the device memory.
    dqDesc
        Input. Descriptor for queries and dqueries sequence data.
    dqueries
        Output. Device pointer to gradients of the loss function computed with respect to queries vectors.
    queries
        Input. Pointer to queries data in the device memory. This is the same input as in cudnnMultiHeadAttnForward().
    dkDesc
        Input. Descriptor for keys and dkeys sequence data.
    dkeys
        Output. Device pointer to gradients of the loss function computed with respect to keys vectors.
    keys
        Input. Pointer to keys data in the device memory. This is the same input as in cudnnMultiHeadAttnForward().
    dvDesc
        Input. Descriptor for values and dvalues sequence data.
    dvalues
        Output. Device pointer to gradients of the loss function computed with respect to values vectors.
    values
        Input. Pointer to values data in the device memory. This is the same input as in cudnnMultiHeadAttnForward().
    weightSizeInBytes
        Input. Size of the weight buffer in bytes where all multi-head attention trainable parameters are stored.
    weights
        Input. Pointer to the weight buffer in device memory.
    workSpaceSizeInBytes
        Input. Size of the work-space buffer in bytes used for temporary API storage.
    workSpace
        Input/Output. Pointer to the work-space buffer in device memory.
    reserveSpaceSizeInBytes
        Input. Size of the reserve-space buffer in bytes used for data exchange between forward and backward (gradient) 
        API calls. This parameter should be zero in the inference mode and non-zero in the training mode.
    reserveSpace
        Input/Output. Pointer to the reserve-space buffer in device memory. This argument should be NULL in inference mode 
        and non-NULL in the training mode.
    """

    status = _libcudnn.cudnnMultiHeadAttnBackwardData(handle, attnDesc, loWinIdx.ctypes.data, hiWinIdx.ctypes.data,
                                                      devSeqLengthsDQDO, devSeqLengthsDKDV,
                                                      doDesc, dout, dqDesc, dqueries, queries, dkDesc, dkeys, keys, dvDesc, dvalues, values,
                                                      weightSizeInBytes, weights, workSpaceSizeInBytes, workSpace, reserveSpaceSizeInBytes,
                                                      reserveSpace)
    cudnnCheckStatus(status)


_libcudnn.cudnnMultiHeadAttnBackwardWeights.restype = int
_libcudnn.cudnnMultiHeadAttnBackwardWeights.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                                                        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                                                        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                                                        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                                                        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t,
                                                        ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p]


def cudnnMultiHeadAttnBackwardWeights(handle, attnDesc, addGrad,
                                      qDesc, queries, kDesc, keys, vDesc, values, doDesc, dout,
                                      weightSizeInBytes, weights, dweights,
                                      workSpaceSizeInBytes, workSpace,
                                      reserveSpaceSizeInBytes, reserveSpace):
    """"
    This function computes exact, first-order derivatives of the multi-head attention block with respect to its trainable parameters: 
    projection weights and projection biases. If y=F(w) is a vector-valued function that represents the multi-head attention layer 
    and it takes some vector x ϵ ℝ n of flatten weights or biases as an input (with all other parameters and inputs fixed), and 
    outputs vector y ϵ ℝ m , then cudnnMultiHeadAttnBackwardWeights() computes the result of ∂ y i / ∂ x j T δ out where δ out is the 
    m × 1 gradient of the loss function with respect to multi-head attention outputs. The δ out gradient is back propagated through 
    prior layers of the deep learning model. ∂ y i / ∂ x j is the m × n Jacobian matrix of F(w). The δ out input is supplied via the 
    dout argument.
    Parameters
    ----------
    handle
        Input. The current cuDNN context handle.
    attnDesc
        Input. A previously initialized attention descriptor.
    addGrad
        Input. Weight gradient output mode.
    qDesc
        Input. Descriptor for queries sequence data.
    queries
        Input. Pointer to queries data in the device memory.
    kDesc
        Input. Descriptor for keys sequence data.
    keys
        Input. Pointer to keys data in the device memory.
    vDesc
        Input. Descriptor for values sequence data.
    values
        Input. Pointer to values data in the device memory.
    doDesc
        Input. Descriptor for the δ out gradients (vectors of partial derivatives of the loss function with respect to the multi-head attention outputs).
    dout
        Pointer to δ out gradient data in the device memory.
    weightSizeInBytes
        Input. Size of the weight buffer in bytes where all multi-head attention trainable parameters are stored.
    weights
        Input. Pointer to the weight buffer in device memory.
    dweights
        Output. Address of the weight gradient buffer in the device memory.
    workSpaceSizeInBytes
        Input. Size of the work-space buffer in bytes used for temporary API storage.
    workSpace
        Input/Output. Pointer to the work-space buffer in device memory.
    reserveSpaceSizeInBytes
        Input. Size of the reserve-space buffer in bytes used for data exchange between forward and backward (gradient) 
        API calls.
    reserveSpace
        Input/Output. Pointer to the reserve-space buffer in device memory.
    """

    status = _libcudnn.cudnnMultiHeadAttnBackwardWeights(handle, attnDesc, addGrad,
                                                         qDesc, queries, kDesc, keys, vDesc, values, doDesc, dout,
                                                         weightSizeInBytes, weights, dweights,
                                                         workSpaceSizeInBytes, workSpace,
                                                         reserveSpaceSizeInBytes, reserveSpace)
    cudnnCheckStatus(status)


cudnnNormOps = {
    'CUDNN_NORM_OPS_NORM': 0,
    'CUDNN_NORM_OPS_NORM_ACTIVATION': 1,
    'CUDNN_NORM_OPS_NORM_ADD_ACTIVATION': 2
}

cudnnNormAlgo = {
    'CUDNN_NORM_ALGO_STANDARD': 0,
    'CUDNN_NORM_ALGO_PERSIST': 1
}

cudnnNormMode = {
    'CUDNN_NORM_PER_ACTIVATION': 0,
    'CUDNN_NORM_PER_CHANNEL': 1
}

_libcudnn.cudnnNormalizationForwardTraining.restype = int
_libcudnn.cudnnNormalizationForwardTraining.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                                                        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                                                        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                                                        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_double,
                                                        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                                                        ctypes.c_double, ctypes.c_void_p, ctypes.c_void_p,
                                                        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                                                        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                                                        ctypes.c_size_t, ctypes.c_void_p, ctypes.c_size_t,
                                                        ctypes.c_int]


def cudnnNormalizationForwardTraining(handle, mode, normOps, algo, alpha, beta,
                                      xDesc, xData, normScaleBiasDesc, normScale, normBias,
                                      exponentialAverageFactor, normMeanVarDesc,
                                      resultRunningMean, resultRunningVariance,
                                      epsilon, resultSaveMean, resultSaveInvVariance,
                                      activationDesc, zDesc, zData, yDesc, yData,
                                      workspace, workSpaceSizeInBytes,
                                      reserveSpace, reserveSpaceSizeInBytes,
                                      groupCnt):
    '''
    This function performs the forward normalization layer computation for the training phase. 
    Depending on mode, different normalization operations will be performed.
    Parameters
    ----------
    handle
        Input. Handle to a previously created cuDNN library descriptor. For more information, see cudnnHandle_t.
    mode
        Input. Mode of operation (per-channel or per-activation). For more information, see cudnnNormMode_t.
    normOps
        Input. Mode of post-operative. Currently CUDNN_NORM_OPS_NORM_ACTIVATION and CUDNN_NORM_OPS_NORM_ADD_ACTIVATION are only supported in the NHWC layout. For more information, see cudnnNormOps_t. This input can be used to set this function to perform either only the normalization, or normalization followed by activation, or normalization followed by element-wise addition and then activation.
    algo
        Input. Algorithm to be performed. For more information, see cudnnNormAlgo_t.
    *alpha, *beta
        Inputs. Pointers to scaling factors (in host memory) used to blend the layer output value with prior value in the destination tensor as follows:
        dstValue = alpha[0]*resultValue + beta[0]*priorDstValue
    xDesc, yDesc
        Input. Handles to the previously initialized tensor descriptors.
    *xData
        Input. Data pointer to GPU memory associated with the tensor descriptor xDesc, for the layer’s x input data.
    *yData
        Output. Data pointer to GPU memory associated with the tensor descriptor yDesc, for the y output of the normalization layer.
    zDesc, *zData
        Input. Tensor descriptors and pointers in device memory for residual addition to the result of the normalization operation, prior to the activation. zDesc and *zData are optional and are only used when normOps is CUDNN_NORM_OPS_NORM_ADD_ACTIVATION, otherwise the user may pass NULL. When in use, z should have exactly the same dimension as xData and the final output yData. For more information, see cudnnTensorDescriptor_t.
    normScaleBiasDesc, normScale, normBias
        Inputs. Tensor descriptors and pointers in device memory for the normalization scale and bias parameters (in the original paper bias is referred to as beta and scale as gamma). The dimensions for the tensor descriptor are dependent on the normalization mode.
    exponentialAverageFactor
        Input. Factor used in the moving average computation as follows:
        runningMean = runningMean*(1-factor) + newMean*factor
    normMeanVarDesc
        Inputs. Tensor descriptor used for following tensors: resultRunningMean, resultRunningVariance, resultSaveMean, resultSaveInvVariance.
    *resultRunningMean, *resultRunningVariance
        Inputs/Outputs. Pointers to the running mean and running variance data. Both these pointers can be NULL but only at the same time. The value stored in resultRunningVariance (or passed as an input in inference mode) is the sample variance and is the moving average of variance[x] where the variance is computed either over batch or spatial+batch dimensions depending on the mode. If these pointers are not NULL, the tensors should be initialized to some reasonable values or to 0.
    epsilon
        Input. Epsilon value used in the normalization formula. Its value should be equal to or greater than zero.
    *resultSaveMean, *resultSaveInvVariance
        Outputs. Optional cache parameters containing saved intermediate results computed during the forward pass. For this to work correctly, the layer's x and normScale, normBias data has to remain unchanged until this backward function is called. Note that both these parameters can be NULL but only at the same time. It is recommended to use this cache since the memory overhead is relatively small.
    activationDesc
        Input. The tensor descriptor for the activation operation. When the normOps input is set to either CUDNN_NORM_OPS_NORM_ACTIVATION or CUDNN_NORM_OPS_NORM_ADD_ACTIVATION then this activation is used, otherwise the user may pass NULL.
    *workspace, workSpaceSizeInBytes
        Inputs. *workspace is a pointer to the GPU workspace, and workSpaceSizeInBytes is the size of the workspace. When *workspace is not NULL and *workSpaceSizeInBytes is large enough, and the tensor layout is NHWC and the data type configuration is supported, then this function will trigger a semi-persistent NHWC kernel for normalization. The workspace is not required to be clean. Also, the workspace does not need to remain unchanged between the forward and backward passes.
    *reserveSpace
        Input. Pointer to the GPU workspace for the reserveSpace.
    reserveSpaceSizeInBytes
        Input. The size of the reserveSpace. Must be equal or larger than the amount required by cudnnGetNormalizationTrainingReserveSpaceSize().
    groutCnt
        Input. Only support 1 for now.
    '''

    dataType = cudnnGetTensor4dDescriptor(xDesc)[0]
    if dataType == cudnnDataType['CUDNN_DATA_DOUBLE']:
        alphaRef = ctypes.byref(ctypes.c_double(alpha))
        betaRef = ctypes.byref(ctypes.c_double(beta))
    else:
        alphaRef = ctypes.byref(ctypes.c_float(alpha))
        betaRef = ctypes.byref(ctypes.c_float(beta))
    status = _libcudnn.cudnnNormalizationForwardTraining(handle, mode, normOps, algo, alphaRef, betaRef,
                                                         xDesc, xData, normScaleBiasDesc, normScale, normBias,
                                                         exponentialAverageFactor, normMeanVarDesc,
                                                         resultRunningMean, resultRunningVariance,
                                                         epsilon, resultSaveMean, resultSaveInvVariance,
                                                         activationDesc, zDesc, zData, yDesc, yData,
                                                         workspace, workSpaceSizeInBytes,
                                                         reserveSpace, reserveSpaceSizeInBytes,
                                                         groupCnt)
    cudnnCheckStatus(status)


_libcudnn.cudnnNormalizationForwardInference.restype = int
_libcudnn.cudnnNormalizationForwardInference.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                                                         ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                                                         ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                                                         ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                                                         ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                                                         ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                                                         ctypes.c_void_p, ctypes.c_double, ctypes.c_int]


def cudnnNormalizationForwardInference(handle, mode, normOps, algo, alpha, beta,
                                       xDesc, x, normScaleBiasDesc, normScale, normBias,
                                       normMeanVarDesc, estimatedMean, estimatedVariance,
                                       zDesc, z, activationDesc, yDesc, y, epsilon, groupCnt):
    '''
    This function performs the forward normalization layer computation for the inference phase.
    Parameters
    ----------
    handle
        Input. Handle to a previously created cuDNN library descriptor. For more information, see cudnnHandle_t.
    mode
        Input. Mode of operation (per-channel or per-activation). For more information, see cudnnNormMode_t.
    normOps
        Input. Mode of post-operative. Currently, CUDNN_NORM_OPS_NORM_ACTIVATION and CUDNN_NORM_OPS_NORM_ADD_ACTIVATION are not supported.
    algo
        Input. Algorithm to be performed. For more information, see cudnnNormAlgo_t.
    alpha, beta
        Inputs. Pointers to scaling factors (in host memory) used to blend the layer output value with prior value in the destination tensor as follows:
        dstValue = alpha[0]*resultValue + beta[0]*priorDstValue
    xDesc, yDesc
        Input. Handles to the previously initialized tensor descriptors.
    *x
        Input. Data pointer to GPU memory associated with the tensor descriptor xDesc, for the layer’s x input data.
    *y
        Output. Data pointer to GPU memory associated with the tensor descriptor yDesc, for the y output of the normalization layer.
    zDesc, *z
        Input. Tensor descriptors and pointers in device memory for residual addition to the result of the normalization operation, prior to the activation. zDesc and *z are optional and are only used when normOps is CUDNN_NORM_OPS_NORM_ADD_ACTIVATION, otherwise users may pass NULL. When in use, z should have exactly the same dimension as x and the final output y. For more information, see cudnnTensorDescriptor_t.
     normScaleBiasDesc, normScale, normBias
        Inputs. Tensor descriptors and pointers in device memory for the normalization scale and bias parameters (in the original paper bias is referred to as beta and scale as gamma).
    normMeanVarDesc, estimatedMean, estimatedVariance
        Inputs. Mean and variance tensors and their tensor descriptors. The estimatedMean and estimatedVariance inputs, accumulated during the training phase from the cudnnNormalizationForwardTraining() call, should be passed as inputs here.
    activationDesc
        Input. Descriptor for the activation operation. When the normOps input is set to either CUDNN_NORM_OPS_NORM_ACTIVATION or CUDNN_NORM_OPS_NORM_ADD_ACTIVATION then this activation is used, otherwise the user may pass NULL. Since normOps is only supported for CUDNN_NORM_OPS_NORM, we can set these to NULL for now.
    epsilon
        Input. Epsilon value used in the normalization formula. Its value should be equal to or greater than zero.
    groutCnt
        Input. Only support 1 for now.
    '''

    status = _libcudnn.cudnnNormalizationForwardInference(handle, mode, normOps, algo, alpha, beta,
                                                          xDesc, x, normScaleBiasDesc, normScale, normBias,
                                                          normMeanVarDesc, estimatedMean, estimatedVariance,
                                                          zDesc, z, activationDesc, yDesc, y, epsilon, groupCnt)
    cudnnCheckStatus(status)


_libcudnn.cudnnNormalizationBackward.restype = int
_libcudnn.cudnnNormalizationBackward.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                                                 ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                                                 ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                                                 ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                                                 ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                                                 ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                                                 ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                                                 ctypes.c_void_p, ctypes.c_void_p, ctypes.c_double,
                                                 ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                                                 ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t,
                                                 ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]


def cudnnNormalizationBackward(handle, mode, normOps, algo, alphaDataDiff, betaDataDiff,
                               alphaParamDiff, betaParamDiff, xDesc, xData, yDesc, yData,
                               dyDesc, dyData, dzDesc, dzData, dxDesc, dxData,
                               dNormScaleBiasDesc, normScaleData, normBiasData,
                               dNormScaleData, dNormBiasData, epsilon,
                               normMeanVarDesc, savedMean, savedInvVariance,
                               activationDesc, workSpace, workSpaceSizeInBytes,
                               reserveSpace, reserveSpaceSizeInBytes, groupCnt):
    '''
    This function performs backward normalization layer computation that is specified by mode.
    Parameters
    ----------
    handle
        Input. Handle to a previously created cuDNN library descriptor. For more information, see cudnnHandle_t.
    mode
        Input. Mode of operation (per-channel or per-activation). For more information, see cudnnNormMode_t.
    normOps
        Input. Mode of post-operative. Currently CUDNN_NORM_OPS_NORM_ACTIVATION and CUDNN_NORM_OPS_NORM_ADD_ACTIVATION are only supported in the NHWC layout. For more information, see cudnnNormOps_t. This input can be used to set this function to perform either only the normalization, or normalization followed by activation, or normalization followed by element-wise addition and then activation.
    algo
        Input. Algorithm to be performed. For more information, see cudnnNormAlgo_t.
    *alphaDataDiff, *betaDataDiff
        Inputs. Pointers to scaling factors (in host memory) used to blend the gradient output dx with a prior value in the destination tensor as follows:
        dstValue = alpha[0]*resultValue + beta[0]*priorDstValue
    *alphaParamDiff, *betaParamDiff
        Inputs. Pointers to scaling factors (in host memory) used to blend the gradient outputs dNormScaleData and dNormBiasData with prior values in the destination tensor as follows:
        dstValue = alpha[0]*resultValue + beta[0]*priorDstValue
    xDesc, *xData, yDesc, *yData, dyDesc, *dyData
        Inputs. Tensor descriptors and pointers in the device memory for the layer's x data, backpropagated gradient input dy, the original forward output y data. yDesc and yData are not needed if normOps is set to CUDNN_NORM_OPS_NORM, users may pass NULL. For more information, see cudnnTensorDescriptor_t.
    dzDesc, *dzData, dxDesc, *dxData
        Outputs. Tensor descriptors and pointers in the device memory for the computed gradient output dz and dx. dzDesc and *dzData is not needed when normOps is CUDNN_NORM_OPS_NORM or CUDNN_NORM_OPS_NORM_ACTIVATION, users may pass NULL. For more information, see cudnnTensorDescriptor_t.
    dNormScaleBiasDesc
        Input. Shared tensor descriptor for the following six tensors: normScaleData, normBiasData, dNormScaleData, and dNormBiasData. The dimensions for this tensor descriptor are dependent on normalization mode.
    *normScaleData
        Input. Pointer in the device memory for the normalization scale parameter (in the original paper the quantity scale is referred to as gamma).
    *normBiasData
        Input. Pointers in the device memory for the normalization bias parameter (in the original paper bias is referred to as beta). This parameter is used only when activation should be performed.
    *dNormScaleData, dNormBiasData
        Inputs. Pointers in the device memory for the gradients of normScaleData and normBiasData, respectively.
    epsilon
        Input. Epsilon value used in normalization formula. Its value should be equal to or greater than zero. The same epsilon value should be used in forward and backward functions.
    normMeanVarDesc
        Input. Shared tensor descriptor for the following tensors: savedMean and savedInvVariance. The dimensions for this tensor descriptor are dependent on normalization mode.
    *savedMean, *savedInvVariance
        Inputs. Optional cache parameters containing saved intermediate results computed during the forward pass. For this to work correctly, the layer's x and normScaleData, normBiasData data has to remain unchanged until this backward function is called. Note that both these parameters can be NULL but only at the same time. It is recommended to use this cache since the memory overhead is relatively small.
    activationDesc
        Input. Descriptor for the activation operation. When the normOps input is set to either CUDNN_NORM_OPS_NORM_ACTIVATION or CUDNN_NORM_OPS_NORM_ADD_ACTIVATION then this activation is used, otherwise the user may pass NULL.
    workspace
        Input. Pointer to the GPU workspace.
    workSpaceSizeInBytes
        Input. The size of the workspace. It must be large enough to trigger the fast NHWC semi-persistent kernel by this function.
    *reserveSpace
        Input. Pointer to the GPU workspace for the reserveSpace.
    reserveSpaceSizeInBytes
        Input. The size of the reserveSpace. It must be equal or larger than the amount required by cudnnGetNormalizationTrainingReserveSpaceSize().
    groutCnt
        Input. Only support 1 for now.
    '''
    status = _libcudnn.cudnnNormalizationBackward(handle, mode, normOps, algo, alphaDataDiff, betaDataDiff,
                                                  alphaParamDiff, betaParamDiff, xDesc, xData, yDesc, yData,
                                                  dyDesc, dyData, dzDesc, dzData, dxDesc, dxData,
                                                  dNormScaleBiasDesc, normScaleData, normBiasData,
                                                  dNormScaleData, dNormBiasData, epsilon,
                                                  normMeanVarDesc, savedMean, savedInvVariance,
                                                  activationDesc, workSpace, workSpaceSizeInBytes,
                                                  reserveSpace, reserveSpaceSizeInBytes, groupCnt)
    cudnnCheckStatus(status)


_libcudnn.cudnnGetNormalizationBackwardWorkspaceSize.restype = int
_libcudnn.cudnnGetNormalizationBackwardWorkspaceSize.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                                                                 ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                                                                 ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                                                                 ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                                                                 ctypes.c_void_p, ctypes.c_int]


def cudnnGetNormalizationBackwardWorkspaceSize(handle,
                                               mode,
                                               normOps,
                                               algo,
                                               xDesc,
                                               yDesc,
                                               dyDesc,
                                               dzDesc,
                                               dxDesc,
                                               dNormScaleBiasDesc,
                                               activationDesc,
                                               normMeanVarDesc,
                                               groupCnt):
    sizeInBytes = ctypes.c_size_t()
    status = _libcudnn.cudnnGetNormalizationBackwardWorkspaceSize(handle,
                                                                  mode,
                                                                  normOps,
                                                                  algo,
                                                                  xDesc,
                                                                  yDesc,
                                                                  dyDesc,
                                                                  dzDesc,
                                                                  dxDesc,
                                                                  dNormScaleBiasDesc,
                                                                  activationDesc,
                                                                  normMeanVarDesc,
                                                                  ctypes.byref(sizeInBytes),
                                                                  groupCnt)
    cudnnCheckStatus(status)
    return sizeInBytes


_libcudnn.cudnnGetNormalizationForwardTrainingWorkspaceSize.restype = int
_libcudnn.cudnnGetNormalizationForwardTrainingWorkspaceSize.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                                                                        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                                                                        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                                                                        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int]


def cudnnGetNormalizationForwardTrainingWorkspaceSize(handle,
                                                      mode,
                                                      normOps,
                                                      algo,
                                                      xDesc,
                                                      zDesc,
                                                      yDesc,
                                                      normScaleBiasDesc,
                                                      activationDesc,
                                                      normMeanVarDesc,
                                                      groupCnt):
    sizeInBytes = ctypes.c_size_t()
    status = _libcudnn.cudnnGetNormalizationForwardTrainingWorkspaceSize(handle,
                                                                         mode,
                                                                         normOps,
                                                                         algo,
                                                                         xDesc,
                                                                         zDesc,
                                                                         yDesc,
                                                                         normScaleBiasDesc,
                                                                         activationDesc,
                                                                         normMeanVarDesc,
                                                                         ctypes.byref(sizeInBytes),
                                                                         groupCnt)
    cudnnCheckStatus(status)
    return sizeInBytes


_libcudnn.cudnnGetNormalizationTrainingReserveSpaceSize.restype = int
_libcudnn.cudnnGetNormalizationTrainingReserveSpaceSize.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                                                                    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                                                                    ctypes.c_void_p, ctypes.c_int]


def cudnnGetNormalizationTrainingReserveSpaceSize(handle,
                                                  mode,
                                                  normOps,
                                                  algo,
                                                  activationDesc,
                                                  xDesc,
                                                  groupCnt):
    sizeInBytes = ctypes.c_size_t()
    status = _libcudnn.cudnnGetNormalizationTrainingReserveSpaceSize(handle,
                                                                     mode,
                                                                     normOps,
                                                                     algo,
                                                                     activationDesc,
                                                                     xDesc,
                                                                     ctypes.byref(sizeInBytes),
                                                                     groupCnt)
    cudnnCheckStatus(status)
    return sizeInBytes


def cudnnBatchNormalizationBackward(handle, mode, alpha_data_diff, beta_data_diff, alpha_param_diff, beta_param_diff,
                                    x_desc, x, dy_desc, dy, dx_desc, dx, bn_scale_bias_diff_desc, bn_scale,
                                    result_bn_scale_diff, result_bn_bias_diff, epsilon, saved_mean, saved_inv_variance):
    """
    This function performs the backward batch normalization layer computation.
    This layer is based on the paper Batch Normalization: Accelerating
    Deep Network Training by Reducing Internal Covariate Shift, S. Ioffe, C. Szegedy, 2015. .

    The epsilon value has to be the same during training, backpropagation, and inference.
    -----------------
    handle
    Input. Handle to a previously created cuDNN library descriptor.
    mode
    Input. Mode of operation (spatial or per-activation).
    alpha_data_diff, beta_data_diff
    Inputs. Pointers to scaling factors (in host memory) used to blend the gradient
    output dx with a prior value in the destination tensor as follows:
            dstValue = alpha_data_diff[0]*resultValue + beta_data_diff[0]*priorDstValue
    alpha_param_diff, *beta_param_diff
    Inputs. Pointers to scaling factors (in host memory) used to blend the gradient outputs
    result_bn_scale_diff and result_bn_bias_diff with prior values in the destination tensor as follows:
            dstValue = alpha_param_diff[0]*resultValue + beta_param_diff[0]*priorDstValue
    x_desc, dx_desc, dy_desc
    Inputs. Handles to the previously initialized tensor descriptors.
    x
    Input. Data pointer to GPU memory associated with the tensor descriptor x_desc, for the layer’s x data.
    dy
    Inputs. Data pointer to GPU memory associated with the tensor descriptor dy_desc,
    for the backpropagated differential dy input.
    dx
    Outputs. Data pointer to GPU memory associated with the tensor descriptor dx_desc,
    for the resulting differential output with respect to x.
    bn_scale_bias_diff_desc
    Input. Shared tensor descriptor for the following five tensors: bn_scale, result_bn_scale_diff,
    result_bn_bias_diff, saved_mean, saved_inv_variance. The dimensions for this tensor descriptor
    are dependent on normalization mode.

    *bn_scale
    Input. Pointer in the device memory for the batch normalization scale parameter
    (in the original paper the quantity scale is referred to as gamma).
    Note: The bn_bias parameter is not needed for this layer's computation.

    result_bn_scale_diff, result_bn_bias_diff
    Outputs. Pointers in device memory for the resulting scale and bias differentials
    computed by this routine. Note that these scale and bias gradients are weight gradients
    specific to this batch normalization operation, and by definition are not backpropagated.

    epsilon
    Input. Epsilon value used in batch normalization formula.
    Its value should be equal to or greater than the value defined for
    CUDNN_BN_MIN_EPSILON in cudnn.h. The same epsilon value should be
    used in forward and backward functions.
    *saved_mean, *saved_inv_variance
    Inputs. Optional cache parameters containing saved intermediate results that were
    computed during the forward pass. For this to work correctly, the layer's x and
    bn_scale data have to remain unchanged until this backward function is called.
    """

    data_type = cudnnGetTensor4dDescriptor(x_desc)[0]
    if data_type == cudnnDataType['CUDNN_DATA_DOUBLE']:
        alpha_data_ref = ctypes.byref(ctypes.c_double(alpha_data_diff))
        beta_data_ref = ctypes.byref(ctypes.c_double(beta_data_diff))
        alpha_param_ref = ctypes.byref(ctypes.c_double(alpha_param_diff))
        beta_param_ref = ctypes.byref(ctypes.c_double(beta_param_diff))

    else:
        alpha_data_ref = ctypes.byref(ctypes.c_float(alpha_data_diff))
        beta_data_ref = ctypes.byref(ctypes.c_float(beta_data_diff))
        alpha_param_ref = ctypes.byref(ctypes.c_float(alpha_param_diff))
        beta_param_ref = ctypes.byref(ctypes.c_float(beta_param_diff))

    status = _libcudnn.cudnnBatchNormalizationBackward(handle, mode, alpha_data_ref,
                                                       beta_data_ref, alpha_param_ref, beta_param_ref, x_desc, x,
                                                       dy_desc, dy,
                                                       dx_desc, dx,
                                                       bn_scale_bias_diff_desc, bn_scale, result_bn_scale_diff,
                                                       result_bn_bias_diff, epsilon,
                                                       saved_mean, saved_inv_variance)

    cudnnCheckStatus(status)


_libcudnn.cudnnBatchNormalizationForwardInference.restype = int
_libcudnn.cudnnBatchNormalizationForwardInference.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p,
                                                              ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                                                              ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                                                              ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                                                              ctypes.c_void_p, ctypes.c_double]


def cudnnBatchNormalizationForwardInference(handle, mode, alpha, beta, x_desc, x, y_desc, y,
                                            bn_scale_bias_mean_var_desc, bn_scale, bn_bias, estimated_mean,
                                            estimated_variance, epsilon):
    """
    This function performs the forward batch normalization layer
    computation for the inference phase. This layer is based on the
    paper Batch Normalization: Accelerating Deep Network Training by
    Reducing Internal Covariate Shift, S. Ioffe, C. Szegedy, 2015.
    -----------------------------------
    handle
    Input. Handle to a previously created cuDNN library descriptor.
    For more information, see cudnnHandle_t.
    mode
    Input. Mode of operation (spatial or per-activation).
    For more information, see cudnnBatchNormMode_t.
    alpha, beta
    Inputs. Pointers to scaling factors (in host memory) used to blend the layer
    output value with prior value in the destination tensor as follows:
        dstValue = alpha[0]*resultValue + beta[0]*priorDstValue
    x_desc, y_desc
    Input. Handles to the previously initialized tensor descriptors.
    x
    Input. Data pointer to GPU memory associated with the tensor
    descriptor x_desc, for the layer’s x input data.
    y
    Output. Data pointer to GPU memory associated with the tensor
    descriptor y_desc, for the youtput of the batch normalization layer.
    bn_scale_bias_mean_var_desc, bn_scale, bn_bias
    Inputs. Tensor descriptors and pointers in device memory for the
    batch normalization scale and bias parameters (in the original paper
    bias is referred to as beta and scale as gamma).
    estimated_mean, estimated_variance
    Inputs. Mean and variance tensors (these have the same descriptor as
    the bias and scale). The result_running_mean and result_running_variance,
    accumulated during the training phase from the cudnnBatchNormalizationForwardTraining()
    call, should be passed as inputs here.
    epsilon
    Input. Epsilon value used in the batch normalization formula. Its value
    should be equal to or greater than the value defined for CUDNN_BN_MIN_EPSILON in cudnn.h.

    """

    data_type = cudnnGetTensor4dDescriptor(x_desc)[0]
    if data_type == cudnnDataType['CUDNN_DATA_DOUBLE']:
        alpha_ref = ctypes.byref(ctypes.c_double(alpha))
        beta_ref = ctypes.byref(ctypes.c_double(beta))

    else:
        alpha_ref = ctypes.byref(ctypes.c_float(alpha))
        beta_ref = ctypes.byref(ctypes.c_float(beta))

    status = _libcudnn.cudnnBatchNormalizationForwardInference(handle, mode, alpha_ref, beta_ref,
                                                               x_desc, x, y_desc, y, bn_scale_bias_mean_var_desc,
                                                               bn_scale,
                                                               bn_bias,
                                                               estimated_mean, estimated_variance, epsilon)

    cudnnCheckStatus(status)


_libcudnn.cudnnBatchNormalizationForwardTraining.restype = int
_libcudnn.cudnnBatchNormalizationForwardTraining.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p,
                                                             ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                                                             ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                                                             ctypes.c_void_p, ctypes.c_void_p, ctypes.c_double,
                                                             ctypes.c_void_p, ctypes.c_void_p, ctypes.c_double,
                                                             ctypes.c_void_p, ctypes.c_void_p]


def cudnnBatchNormalizationForwardTraining(handle, mode, alpha, beta, x_desc, x, y_desc, y, bn_scale_bias_mean_var_desc,
                                           bn_scale, bn_bias, exponential_average_factor, result_running_mean,
                                           result_running_variance, epsilon, result_save_mean,
                                           result_save_inv_variance):
    """
    This function performs the forward batch normalization layer computation
    for the training phase. This layer is based on the paper
    Batch Normalization: Accelerating Deep Network Training by Reducing
    Internal Covariate Shift, S. Ioffe, C. Szegedy, 2015.
    handle: Handle to a previously created cuDNN library descriptor.
        For more information, see cudnnHandle_t.
    mode: Mode of operation (spatial or per-activation).
        For more information, see cudnnBatchNormMode_t.
    alpha, beta: Inputs. Pointers to scaling factors (in host memory) used to
        blend the layer output value with prior value in the destination tensor as follows:
        dstValue = alpha[0]*resultValue + beta[0]*priorDstValue
    x_desc, y_desc: Tensor descriptors and pointers in device memory for the
        layer's x and y data. For more information, see cudnnTensorDescriptor_t.
    *x: Input. Data pointer to GPU memory associated with the tensor descriptor x_desc,
        for the layer’s x input data.
    *y: Output. Data pointer to GPU memory associated with the tensor descriptor y_desc,
        for the y output of the batch normalization layer.
    bn_scale_bias_mean_var_desc: Shared tensor descriptor desc for the secondary tensor
        that was derived by cudnnDeriveBNTensorDescriptor().
    bn_scale, bn_bias: Inputs. Pointers in device memory for the batch normalization
        scale and bias parameters (in the original paper bias is referred to
        as beta and scale as gamma).

    exponential_average_factor: Input. Factor used in the moving average computation as follows:
        runningMean = runningMean*(1-factor) + newMean*factor

    result_running_mean, result_running_variance: Inputs/Outputs. Running mean and variance tensors
         (these have the same descriptor as the bias and scale). Both of these pointers
         can be NULL but only at the same time. The value stored in result_running_variance
         (or passed as an input in inference mode) is the sample variance and is the
         moving average of variance[x] where the variance is computed either over batch
         or spatial+batch dimensions depending on the mode.
         If these pointers are not NULL, the tensors should be initialized to some reasonable values or to 0.
    epsilon: Input. Epsilon value used in the batch normalization formula.
         Its value should be equal to or greater than the value defined for
         CUDNN_BN_MIN_EPSILON in cudnn.h (1e-5).
         The same epsilon value should be used in forward and backward functions.
    result_save_mean, result_save_inv_variance: Outputs. Optional cache to save intermediate
         results computed during the forward pass. These buffers can be used to speed up
         the backward pass when supplied to the cudnnBatchNormalizationBackward() function.
 The intermediate results stored in result_save_mean and result_save_inv_variance buffers
         should not be used directly by the user. Depending on the batch normalization mode,
         the results stored in result_save_inv_variance may vary. For the cache to work
         correctly, the input layer data must remain unchanged until the backward function
         is called. Note that both parameters can be NULL but only at the same time.
         In such a case, intermediate statistics will not be saved, and
         cudnnBatchNormalizationBackward() will have to re-compute them. It is recommended
         to use this cache as the memory overhead is relatively small because these tensors
         have a much lower product of dimensions than the data tensors.
    """

    data_type = cudnnGetTensor4dDescriptor(x_desc)[0]
    if data_type == cudnnDataType['CUDNN_DATA_DOUBLE']:
        alpha_ref = ctypes.byref(ctypes.c_double(alpha))
        beta_ref = ctypes.byref(ctypes.c_double(beta))
    else:
        alpha_ref = ctypes.byref(ctypes.c_float(alpha))
        beta_ref = ctypes.byref(ctypes.c_float(beta))
    status = _libcudnn.cudnnBatchNormalizationForwardTraining(handle, mode, alpha_ref, beta_ref,
                                                              x_desc, x, y_desc, y, bn_scale_bias_mean_var_desc,
                                                              bn_scale,
                                                              bn_bias, exponential_average_factor,
                                                              result_running_mean, result_running_variance, epsilon,
                                                              result_save_mean, result_save_inv_variance)

    cudnnCheckStatus(status)


_libcudnn.cudnnActivationForward.restype = int
_libcudnn.cudnnActivationForward.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                                             ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                                             ctypes.c_void_p, ctypes.c_void_p]


def cudnnActivationForward(handle, act_descriptor, alpha, src_desc, src_data, beta, dest_desc, dest_data):
    """"
    Apply activation function.
    This routine applies a specified neuron activation function element-wise over each input
    value.
    In-place operation is allowed for this routine; i.e., src_data and dest_data pointers
    may be equal. However, this requires src_desc and dest_desc descriptors to be
    identical (particularly, the strides of the input and output must match for in-place
    operation to be allowed).
    Parameters
    ----------
    handle : cudnnHandle
        Handle to a previously created cuDNN context.
    act_descriptor : New in this versione
        Enumerant to specify the activation mode.
    alpha: float
        Scaling factor with which every element of the input tensor is multiplied.
    src_desc : cudnnTensor4dDescription
        Handle to the previously initialized input tensor descriptor.
    src_data : void_p
        Data pointer to GPU memory associated with the tensor descriptor
        src_desc.
    beta: float
        Scaling factor which is applied on every element of the output tensor prior
        to adding the result of the activation Note that if beta is zero, the output
        is not read and can contain any uninitialized data (including Nan numbers).
    dest_desc : cudnnTensor4dDescription
        Handle to the previously initialized output tensor descriptor.
    dest_data : void_p
        Data pointer to GPU memory associated with the output tensor descriptor
        dest_desc.
    """

    data_type = cudnnGetTensor4dDescriptor(dest_desc)[0]
    if data_type == cudnnDataType['CUDNN_DATA_DOUBLE']:
        alpha_ref = ctypes.byref(ctypes.c_double(alpha))
        beta_ref = ctypes.byref(ctypes.c_double(beta))
    else:
        alpha_ref = ctypes.byref(ctypes.c_float(alpha))
        beta_ref = ctypes.byref(ctypes.c_float(beta))

    status = _libcudnn.cudnnActivationForward(handle, act_descriptor, alpha_ref, src_desc, src_data,
                                              beta_ref, dest_desc, dest_data)
    cudnnCheckStatus(status)


_libcudnn.cudnnActivationBackward.restype = int
_libcudnn.cudnnActivationBackward.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                                              ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                                              ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                                              ctypes.c_void_p, ctypes.c_void_p]


def cudnnActivationBackward(handle, act_desc, alpha, src_desc, src_data, src_diff_esc, src_diff_data,
                            dest_desc, dest_data, beta, dest_diff_desc, dest_diff_data):
    """"
    Gradient of activation function.
    This routine computes the gradient of a neuron activation function.
    In-place operation is allowed for this routine; i.e., src_data and dest_data
    pointers may be equal and src_diff_data and dest_diff_data pointers may be equal.
    However, this requires the corresponding tensor descriptors to be identical
    (particularly, the strides of the input and output must match for in-place operation
    to be allowed).
    Parameters
    ----------
    handle : cudnnHandle
        Handle to a previously created cuDNN context.
    act_desc : activationdescriptor
    alpha: float
        Scaling factor with which every element of the input tensor is multiplied.
    src_desc : cudnnTensorDescriptor
        Handle to the previously initialized input tensor descriptor.
    src_data : void_p
        Data pointer to GPU memory associated with the tensor descriptor
        src_desc.
    src_diff_esc : cudnnTensorDescriptor
        Handle to the previously initialized input differential tensor descriptor.
    src_diff_data : void_p
        Data pointer to GPU memory associated with the tensor descriptor
        src_diff_data.
    dest_desc : cudnnTensorDescriptor
        Handle to the previously initialized output tensor descriptor.
    dest_data : void_p
        Data pointer to GPU memory associated with the output tensor descriptor
        dest_desc.
    beta: float
        Scaling factor which is applied on every element of the output tensor prior
        to adding the result of the activation gradient. Note that if beta is zero, the
        output is not read and can contain any uninitialized data (including Nan numbers).
    dest_diff_desc : cudnnTensorDescriptor
        Handle to the previously initialized output differential tensor descriptor.
    dest_diff_data : void_p
        Data pointer to GPU memory associated with the output tensor descriptor
        dest_diff_desc.
    """

    data_type = cudnnGetTensor4dDescriptor(dest_desc)[0]
    if data_type == cudnnDataType['CUDNN_DATA_DOUBLE']:
        alpha_ref = ctypes.byref(ctypes.c_double(alpha))
        beta_ref = ctypes.byref(ctypes.c_double(beta))
    else:
        alpha_ref = ctypes.byref(ctypes.c_float(alpha))
        beta_ref = ctypes.byref(ctypes.c_float(beta))
    status = _libcudnn.cudnnActivationBackward(handle, act_desc, alpha_ref, src_desc, src_data,
                                               src_diff_esc, src_diff_data,
                                               dest_desc, dest_data, beta_ref,
                                               dest_diff_desc, dest_diff_data)
    cudnnCheckStatus(status)
