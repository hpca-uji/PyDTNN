"""
Python interface to the NVIDIA NCCL library
"""

import sys
import ctypes
import ctypes.util
from enum import Enum

if sys.platform in ('linux2', 'linux'):
    _libnccl_libname_list = ['libnccl.so']
elif sys.platform == 'darwin':
    _libnccl_libname_list = ['libnccl.dylib']
elif sys.platform == 'win32':
    _libnccl_libname_list = ['libnccl.dll']
else:
    raise NotImplementedError('PyDTNN NCCL: current platform is not yet supported!')

_libnccl = None
for _libnccl_libname in _libnccl_libname_list:
    try:
        _libnccl = ctypes.cdll.LoadLibrary(_libnccl_libname)
    except OSError:
        pass
    else:
        break
if _libnccl is None:
    raise OSError('NCCL library not found')

# NCCL error
_libnccl.ncclGetErrorString.restype = ctypes.c_char_p
_libnccl.ncclGetErrorString.argtypes = [ctypes.c_int]


class NcclError(Exception):
    def __init__(self, status):
        self.status = status

    def __str__(self):
        error = _libnccl.ncclGetErrorString(self.status)
        return f'{error}'


def ncclCheckStatus(status):
    """
    Raise NCCL exception
    Raise an exception corresponding to the specified NCCL error code.
    Parameters
    ----------
    status : int
        NCCL error code
    """

    if status != 0:
        raise NcclError(status)


NCCL_UNIQUE_ID_BYTES = 128


class NcclUniqueId(ctypes.Structure):
    _fields_ = [("internal", ctypes.c_char * NCCL_UNIQUE_ID_BYTES)]


class NcclComm(ctypes.Structure):
    pass


ncclComm_t = ctypes.POINTER(NcclComm)

# Helper functions

# Return the NCCL_VERSION_CODE of the NCCL library in the supplied integer.
# This integer is coded with the MAJOR, MINOR and PATCH level of the
# NCCL library

# ncclResult_t  ncclGetVersion(int *version);
_libnccl.ncclGetVersion.restype = ctypes.c_size_t
_libnccl.ncclGetVersion.argtypes = []


def ncclGetVersion():
    """
    Get NCCL Version.
    """
    version = ctypes.c_int()
    _libnccl.ncclGetVersion(ctypes.byref(version))
    return version.value


# ncclResult_t  ncclGetUniqueId(NcclUniqueId* uniqueId);
_libnccl.ncclGetUniqueId.restype = int
_libnccl.ncclGetUniqueId.argtypes = []


def ncclGetUniqueId():
    """
    Generates an Id to be used in ncclCommInitRank. ncclGetUniqueId should be
    called once and the Id should be distributed to all ranks in the
    communicator before calling ncclCommInitRank.

    Returns
    -------
    unique_id :
    """

    unique_id = NcclUniqueId()
    status = _libnccl.ncclGetUniqueId(ctypes.byref(unique_id))
    ncclCheckStatus(status)
    return unique_id


# ncclResult_t  ncclCommInitRank(ncclComm_t* comm, int nranks, NcclUniqueId comm_id, int rank);
_libnccl.ncclCommInitRank.restype = int
_libnccl.ncclCommInitRank.argtypes = [ctypes.POINTER(ncclComm_t), ctypes.c_int,
                                      NcclUniqueId, ctypes.c_int]


def ncclCommInitRank(nranks, comm_id, rank):
    """
    Creates a new communicator (multi thread/process version).
    rank must be between 0 and nranks-1 and unique within a communicator clique.
    Each rank is associated to a CUDA device, which has to be set before calling
    ncclCommInitRank.
    ncclCommInitRank implicitly synchronizes with other ranks, so it must be
    called by different threads/processes or use ncclGroupStart/ncclGroupEnd.

    Parameters
    ----------
    nranks :
    comm_id :
    rank :

    Returns
    -------
    comm :
    """

    comm = ncclComm_t()
    status = _libnccl.ncclCommInitRank(ctypes.byref(comm), nranks, comm_id, rank)
    ncclCheckStatus(status)
    return comm


# ncclResult_t  ncclCommInitAll(ncclComm_t* comm, int ndev, const int* dev_list);
_libnccl.ncclCommInitAll.restype = int
_libnccl.ncclCommInitAll.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.POINTER(ctypes.c_int)]


def ncclCommInitAll(dev_list):
    """
    Creates a clique of communicators (single process version).
    This is a convenience function to create a single-process communicator clique.
    Returns an array of ndev newly initialized communicators in comm.
    comm should be pre-allocated with size at least ndev*sizeof(ncclComm_t).
    If dev_list is NULL, the first ndev CUDA devices are used.
    Order of dev_list defines user-order of processors within the communicator.

    Parameters
    ----------
    dev_list :

    Returns
    -------
    comm :
    """
    comm = ncclComm_t() * len(dev_list)
    status = _libnccl.ncclCommInitAll(ctypes.byref(comm), len(dev_list), dev_list)
    ncclCheckStatus(status)
    return comm.value


# ncclResult_t  ncclCommDestroy(ncclComm_t comm);
_libnccl.ncclCommDestroy.restype = int
_libnccl.ncclCommDestroy.argtypes = [ncclComm_t]


def ncclCommDestroy(comm):
    """
    Frees resources associated with communicator object, but waits for any operations
    that might still be running on the device.

    Parameters
    ----------
    comm :
    """
    status = _libnccl.ncclCommDestroy(comm)
    ncclCheckStatus(status)


# ncclResult_t  ncclCommAbort(ncclComm_t comm);
_libnccl.ncclCommAbort.restype = int
_libnccl.ncclCommAbort.argtypes = [ctypes.c_void_p]


def ncclCommAbort(comm):
    """
    Frees resources associated with communicator object, but waits for any operations
    that might still be running on the device.

    Parameters
    ----------
    comm :
    """
    status = _libnccl.ncclCommAbort(comm)
    ncclCheckStatus(status)


# ncclResult_t  ncclCommGetAsyncError(ncclComm_t comm, ncclResult_t *async_error);
_libnccl.ncclCommGetAsyncError.restype = int
_libnccl.ncclCommGetAsyncError.argtypes = [ctypes.c_void_p, ctypes.c_void_p]


def ncclCommGetAsyncError(comm):
    """
    Checks whether the comm has encountered any asynchronous errors

    Parameters
    ----------
    comm :
    """
    async_error = ctypes.c_int()
    status = _libnccl.ncclCommGetAsyncError(comm, ctypes.byref(async_error))
    ncclCheckStatus(status)
    return async_error.value


# ncclResult_t  ncclCommCount(const ncclComm_t comm, int* count);
_libnccl.ncclCommCount.restype = int
_libnccl.ncclCommCount.argtypes = [ctypes.c_void_p, ctypes.c_void_p]


def ncclCommCount(comm):
    """
    Gets the number of ranks in the communicator clique

    Parameters
    -------
    comm :

    Returns
    -------
    count :
    """

    count = ctypes.c_int()
    status = _libnccl.ncclCommCount(comm, ctypes.byref(count))
    ncclCheckStatus(status)
    return count.value


# ncclResult_t  ncclCommCuDevice(const ncclComm_t comm, int* device);
_libnccl.ncclCommCuDevice.restype = int
_libnccl.ncclCommCuDevice.argtypes = [ctypes.c_void_p, ctypes.c_void_p]


def ncclCommCuDevice(comm):
    """
    Returns the cuda device number associated with the communicator.

    Parameters
    -------
    comm :

    Returns
    -------
    device :
    """

    device = ctypes.c_int()
    status = _libnccl.ncclCommCuDevice(comm, ctypes.byref(device))
    ncclCheckStatus(status)
    return device.value


# ncclResult_t  ncclCommUserRank(const ncclComm_t comm, int* rank);
_libnccl.ncclCommUserRank.restype = int
_libnccl.ncclCommUserRank.argtypes = [ctypes.c_void_p, ctypes.c_void_p]


def ncclCommUserRank(comm):
    """
    Returns the user-ordered "rank" associated with the communicator.

    Parameters
    -------
    comm :

    Returns
    -------
    rank :
    """

    rank = ctypes.c_int()
    status = _libnccl.ncclCommUserRank(comm, ctypes.byref(rank))
    ncclCheckStatus(status)
    return rank.value


# Reduction operation selector
class RedOp(Enum):
    Sum = 0
    Prod = 1
    Max = 2
    Min = 3
    NumOps = 4


# Data types
class DataType(Enum):
    Int8 = Char = 0
    Uint8 = 1
    Int32 = Int = 2
    Uint32 = 3
    Int64 = 4
    Uint64 = 5
    Float16 = Half = 6
    Float32 = Float = 7
    Float64 = Double = 8
    NumTypes = 9


# Collective communication operations
#
# Collective communication operations must be called separately for each
# communicator in a communicator clique.
#
# They return when operations have been enqueued on the CUDA stream.
#
# Since they may perform inter-CPU synchronization, each call has to be done
# from a different thread or process, or need to use Group Semantics (see
# below).


# ncclResult_t  ncclReduce(const void* sendbuff, void* recvbuff, size_t count, ncclDataType_t datatype,
#     ncclRedOp_t op, int root, ncclComm_t comm, cudaStream_t stream);
_libnccl.ncclReduce.restype = int
_libnccl.ncclReduce.argtypes = [ctypes.c_void_p, ctypes.c_void_p,
                                ctypes.c_ulong, ctypes.c_int, ctypes.c_int, ctypes.c_int,
                                ncclComm_t, ctypes.c_void_p]


def ncclReduce(sendbuff, recvbuff, count, datatype, op, root, comm, stream):
    """
    Reduces data arrays of length count in sendbuff into recvbuff using op operation.
    recvbuff may be NULL on all calls except for root device.
    root is the rank (not the CUDA device) where data will reside after the
    operation is complete.

     In-place operation will happen if sendbuff == recvbuff.

    Parameters
    -----------

    Returns
    -------

    """

    status = _libnccl.ncclReduce(sendbuff, recvbuff, count,
                                 datatype.value, op.value, root, comm, stream)
    ncclCheckStatus(status)


# ncclResult_t  ncclBroadcast(const void* sendbuff, void* recvbuff, size_t count, ncclDataType_t datatype, int root,
#     ncclComm_t comm, cudaStream_t stream);
_libnccl.ncclBroadcast.restype = int
_libnccl.ncclBroadcast.argtypes = [ctypes.c_void_p, ctypes.c_void_p,
                                   ctypes.c_ulong, ctypes.c_int, ctypes.c_int,
                                   ncclComm_t, ctypes.c_void_p]


def ncclBroadcast(sendbuff, recvbuff, count, datatype, root, comm, stream):
    """
    Copies count values from root to all other devices.
    root is the rank (not the CUDA device) where data resides before the
    operation is started.

     In-place operation will happen if sendbuff == recvbuff.

    Parameters
    -----------

    Returns
    -------

    """

    status = _libnccl.ncclBroadcast(sendbuff, recvbuff, count, datatype.value, root, comm, stream)
    ncclCheckStatus(status)


# ncclResult_t  ncclAllReduce(const void* sendbuff, void* recvbuff, size_t count,
#     ncclDataType_t datatype, ncclRedOp_t op, ncclComm_t comm, cudaStream_t stream);
_libnccl.ncclAllReduce.restype = int
_libnccl.ncclAllReduce.argtypes = [ctypes.c_void_p, ctypes.c_void_p,
                                   ctypes.c_ulong, ctypes.c_int, ctypes.c_int,
                                   ncclComm_t, ctypes.c_void_p]


def ncclAllReduce(sendbuff, recvbuff, count, datatype, op, comm, stream):
    """
    Reduces data arrays of length count in sendbuff using op operation, and
    leaves identical copies of result on each recvbuff.

    In-place operation will happen if sendbuff == recvbuff.

    Parameters
    -----------

    Returns
    -------

    """

    status = _libnccl.ncclAllReduce(sendbuff, recvbuff, count,
                                    datatype.value, op.value, comm, stream)
    ncclCheckStatus(status)


# ncclResult_t  ncclReduceScatter(const void* sendbuff, void* recvbuff,
#     size_t recvcount, ncclDataType_t datatype, ncclRedOp_t op, ncclComm_t comm,
#     cudaStream_t stream);
_libnccl.ncclReduceScatter.restype = int
_libnccl.ncclReduceScatter.argtypes = [ctypes.c_void_p, ctypes.c_void_p,
                                       ctypes.c_ulong, ctypes.c_int,
                                       ncclComm_t, ctypes.c_void_p]


def ncclReduceScatter(sendbuff, recvbuff, recvcount, datatype, comm, stream):
    """
    Reduces data in sendbuff using op operation and leaves reduced result
    scattered over the devices so that recvbuff on rank i will contain the i-th
    block of the result.
    Assumes sendcount is equal to nranks*recvcount, which means that sendbuff
    should have a size of at least nranks*recvcount elements.

    In-place operations will happen if recvbuff == sendbuff + rank * recvcount.

    Parameters
    -----------

    Returns
    -------

    """

    status = _libnccl.ncclReduceScatter(sendbuff, recvbuff, recvcount,
                                        datatype.value, comm, stream)
    ncclCheckStatus(status)


# ncclResult_t  ncclAllGather(const void* sendbuff, void* recvbuff, size_t sendcount,
#     ncclDataType_t datatype, ncclComm_t comm, cudaStream_t stream);
_libnccl.ncclAllGather.restype = int
_libnccl.ncclAllGather.argtypes = [ctypes.c_void_p, ctypes.c_void_p,
                                   ctypes.c_ulong, ctypes.c_int,
                                   ncclComm_t, ctypes.c_void_p]


def ncclAllGather(sendbuff, recvbuff, sendcount, datatype, comm, stream):
    """
    Each device gathers sendcount values from other GPUs into recvbuff,
    receiving data from rank i at offset i*sendcount.
    Assumes recvcount is equal to nranks*sendcount, which means that recvbuff
    should have a size of at least nranks*sendcount elements.

    In-place operations will happen if sendbuff == recvbuff + rank * sendcount.

    Parameters
    -----------

    Returns
    -------

    """

    status = _libnccl.ncclAllGather(sendbuff, recvbuff, sendcount,
                                    datatype.value, comm, stream)
    ncclCheckStatus(status)


# ncclResult_t  ncclSend(const void* sendbuff, size_t count, ncclDataType_t datatype, int peer,
#     ncclComm_t comm, cudaStream_t stream);
_libnccl.ncclSend.restype = int
_libnccl.ncclSend.argtypes = [ctypes.c_void_p,
                              ctypes.c_ulong, ctypes.c_int, ctypes.c_int,
                              ncclComm_t, ctypes.c_void_p]


def ncclSend(sendbuff, count, datatype, peer, comm, stream):
    """
    Send data from sendbuff to rank peer.

    Rank peer needs to call ncclRecv with the same datatype and the same count from this
    rank.

    This operation is blocking for the GPU. If multiple ncclSend and ncclRecv operations
    need to progress concurrently to complete, they must be fused within a ncclGroupStart/
    ncclGroupEnd section.

    Parameters
    -----------

    Returns
    -------

    """

    status = _libnccl.ncclSend(sendbuff, count, datatype.value, peer, comm, stream)
    ncclCheckStatus(status)


# ncclResult_t  ncclRecv(void* recvbuff, size_t count, ncclDataType_t datatype, int peer,
#    ncclComm_t comm, cudaStream_t stream);
_libnccl.ncclRecv.restype = int
_libnccl.ncclRecv.argtypes = [ctypes.c_void_p,
                              ctypes.c_int, ctypes.c_int, ctypes.c_int,
                              ncclComm_t, ctypes.c_void_p]


def ncclRecv(recvbuff, count, datatype, peer, comm, stream):
    """
    Receive data from rank peer into recvbuff.

    Rank peer needs to call ncclSend with the same datatype and the same count to this
    rank.

    This operation is blocking for the GPU. If multiple ncclSend and ncclRecv operations
    need to progress concurrently to complete, they must be fused within a ncclGroupStart/
    ncclGroupEnd section.

    Parameters
    -----------

    Returns
    -------

    """

    status = _libnccl.ncclRecv(recvbuff, count, datatype.value, peer, comm, stream)
    ncclCheckStatus(status)


# ncclResult_t  ncclGroupStart();
_libnccl.ncclGroupStart.restype = int
_libnccl.ncclGroupStart.argtypes = []


def ncclGroupStart():
    """
    Start a group call. All calls to NCCL until ncclGroupEnd will be fused into
    a single NCCL operation. Nothing will be started on the CUDA stream until
    ncclGroupEnd.

    Returns
    -------

    """

    status = _libnccl.ncclGroupStart()
    ncclCheckStatus(status)


# ncclResult_t  ncclGroupEnd();
_libnccl.ncclGroupEnd.restype = int
_libnccl.ncclGroupEnd.argtypes = []


def ncclGroupEnd():
    """
    End a group call. Start a fused NCCL operation consisting of all calls since
    ncclGroupStart. Operations on the CUDA stream depending on the NCCL operations
    need to be called after ncclGroupEnd.

    Returns
    -------

    """

    status = _libnccl.ncclGroupEnd()
    ncclCheckStatus(status)
