"""
Python interface to the NVIDIA cuDNN library
"""

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
#  with this program. If not, see <https://www.gnu.org/licenses/>.
# 

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
    raise RuntimeError('unsupported platform')

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

# cudnnAddMode_t is an enumerated type used by cudnnAddTensor() to specify how
# a bias tensor is added to an input/output tensor.
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
cudnnConvolutionMode = {
    'CUDNN_CONVOLUTION': 0,  # In this mode, a convolution operation will be done
    # when applying the filter to the images.
    'CUDNN_CROSS_CORRELATION': 1  # In this mode, a cross-correlation operation will
    # be done when applying the filter to the images.
}

# cudnnConvolutionFwdPreference_t is an enumerated type used by
# cudnnGetConvolutionForwardAlgorithm() to help the choice of the algorithm used for the
# forward convolution.
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

cudnnConvolutionBwdDataPreference = {
    'CUDNN_CONVOLUTION_BWD_DATA_NO_WORKSPACE': 0,
    'CUDNN_CONVOLUTION_BWD_DATA_PREFER_FASTEST': 1,
    'CUDNN_CONVOLUTION_BWD_DATA_SPECIFY_WORKSPACE_LIMIT': 2
}

cudnnConvolutionBwdDataAlgo = {
    'CUDNN_CONVOLUTION_BWD_DATA_ALGO_0': 0,
    'CUDNN_CONVOLUTION_BWD_DATA_ALGO_1': 1,
    'CUDNN_CONVOLUTION_BWD_DATA_ALGO_FFT': 2,
    'CUDNN_CONVOLUTION_BWD_DATA_ALGO_FFT_TILING': 3,
    'CUDNN_CONVOLUTION_BWD_DATA_ALGO_WINOGRAD': 4,
    'CUDNN_CONVOLUTION_BWD_DATA_ALGO_WINOGRAD_NONFUSED': 5,
    'CUDNN_CONVOLUTION_BWD_DATA_ALGO_COUNT': 6
}

cudnnConvolutionBwdFilterPreference = {
    'CUDNN_CONVOLUTION_BWD_FILTER_NO_WORKSPACE': 0,
    'CUDNN_CONVOLUTION_BWD_FILTER_PREFER_FASTEST': 1,
    'CUDNN_CONVOLUTION_BWD_FILTER_SPECIFY_WORKSPACE_LIMIT': 2,
}

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

cudnnBatchNormMode = {
    'CUDNN_BATCHNORM_PER_ACTIVATION': 0,
    'CUDNN_BATCHNORM_SPATIAL': 1,
    'CUDNN_BATCHNORM_SPATIAL_PERSISTENT': 2
}

# cudnnSoftmaxAlgorithm_t is used to select an implementation of the softmax
# function used in cudnnSoftmaxForward() and cudnnSoftmaxBackward().
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
cudnnNanPropagation = {
    'CUDNN_NOT_PROPAGATE_NAN': 0,
    'CUDNN_PROPAGATE_NAN': 1
}
# cudnnActivationMode_t is an enumerated type used to select the neuron activation
# function used in cudnnActivationForward() and cudnnActivationBackward() .
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
    Set all data points of a tensor to a given value : srcDest = alpha.
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
        alpha_ref = ctypes.byref(ctypes.c_double(alpha))
    else:
        alpha_ref = ctypes.byref(ctypes.c_float(alpha))

    status = _libcudnn.cudnnSetTensor(handle, src_desc, src_data, alpha_ref)
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

    return (pad_h.value, pad_w.value, u.value, v.value, upscalex.value, upscaley.value, mode.value,
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
