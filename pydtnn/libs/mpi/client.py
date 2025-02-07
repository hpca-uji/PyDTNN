"""Message Passing Interface (client)"""

# NOTE: Module considerations
#
# Communications are lazily initialized to prevent module imports execution

# FIXME: Implement async communications
# TODO: Optimize lock usage

import enum
import atexit
import functools
import threading
from queue import Empty, SimpleQueue

import numpy as np

from pydtnn import comms
from pydtnn.libs.mpi import comm as mpi_comm


__all__ = (
    "Finalize",
    "IN_PLACE",
    "SUM",
    "COMM_WORLD",
)


def Finalize() -> None:
    """Terminate the MPI execution environment."""


class InPlace(enum.Enum):
    """In-place buffer argument."""
    IN_PLACE = enum.auto()


class Request:
    """Request handler."""

    def wait(self) -> None:
        """Wait for a non-blocking operation to complete."""


class Exception(RuntimeError):
    """Exception class."""


class Intracomm:
    """Intracommunicator."""

    def __init__(self) -> None:
        """Communicator initialization"""
        self._closed = False
        self._lock = threading.Lock()
        self._state = dict[frozenset[mpi_comm.Rank], SimpleQueue[mpi_comm.OperationResponse | None]]()
        atexit.register(self.Disconnect)

    @functools.cached_property
    def size(self) -> int:
        """Communication size"""
        # NOTE: Lazily initialized, prevent module imports execution
        return mpi_comm.get_size()

    @functools.cached_property
    def rank(self) -> mpi_comm.Rank:
        """Communication identifier"""
        # NOTE: Lazily initialized, prevent module imports execution
        return mpi_comm.get_rank()

    def _recive_one(self):
        """Recive one response from communication"""
        assert self._lock.locked(), "Modifing state without lock"
        response = self._comm.get().obj

        match response:
            case mpi_comm.OperationResponse():
                pass
            case _:
                raise RuntimeError(f"Unknown response {response}")

        queue = self._state.setdefault(response.group, SimpleQueue())
        queue.put(response)

    @functools.cached_property
    def _comm(self) -> comms.Communication:
        """Communication connection"""
        # NOTE: Lazily initialized, prevent module imports execution
        comm = comms.Client()
        self._handle_init(comm)
        return comm

    def _handle_init(self, comm: comms.Communication) -> None:
        """Handle communication initialization"""
        # NOTE: Communicator pass to ensure _comm cached_property lock is hit
        msg = mpi_comm.InitRequest(rank=self.rank, size=self.size)
        comm.put(msg)
        comm.get()

    def _handle_finalize(self, comm: comms.Communication) -> None:
        """Handle communication finalization"""
        # NOTE: Communicator pass to ensure _comm cached_property lock is hit
        msg = mpi_comm.FinalizeRequest()
        comm.put(msg)
        comm.get()

    def Disconnect(self) -> None:
        """Disconnect from a communicator."""
        # Atomicly get and remove communicator
        if comm := self.__dict__.pop("_comm", None):
            self._handle_finalize(comm)
            comm.close()

        if self._closed:
            raise Exception()
        self._closed = True

        atexit.unregister(self.Disconnect)
        with self._lock:
            for queue in self._state.values():
                queue.put(None)

    def _get_many(self, group: frozenset[int], size=None):
        """Recive objects to server"""
        if size is None:
            size = self.size

        for _ in range(size):
            yield self._get(group)

    def _get(self, group: frozenset[int]):
        """Recive object to server"""

        with self._lock:
            queue = self._state.setdefault(group, SimpleQueue())

            while True:
                try:
                    response = queue.get_nowait()
                except Empty:
                    self._recive_one()
                else:
                    break

            if queue.empty():
                del self._state[group]

        if response is None:
            raise comms.ResourceClosed()

        return response.obj

    def __del__(self) -> None:
        """Best effort finalizer"""
        try:
            self.Disconnect()
        except:  # noqa: E722
            pass

    def bcast(self, obj, root: mpi_comm.Rank = 0):
        """Broadcast."""
        request = mpi_comm.BroadcastRequest(obj=obj, root=root)
        group = request.response_requirements(size=self.size)

        if self.rank == root:
            self._comm.put(request)
        return self._get(group)

    def barrier(self) -> None:
        """Barrier synchronization."""
        self.allgather(None)

    def allgather(self, obj):
        """Gather to All."""
        request = mpi_comm.AllGatherRequest(obj=obj)
        group = request.response_requirements(size=self.size)

        self._comm.put(request)
        return list(self._get_many(group))

    def allreduce(self, obj, op: mpi_comm.ReduceOperation = mpi_comm.ReduceOperation.SUM):
        """Reduce to All."""
        if op is not mpi_comm.ReduceOperation.SUM:
            raise NotImplementedError("op with not SUM")

        request = mpi_comm.AllReduceRequest(obj=obj, op=op)
        group = request.response_requirements(size=self.size)

        self._comm.put(request)
        return self._get(group)

    def Get_rank(self) -> int:
        """Return the rank of this process in a communicator."""
        return self.rank

    def Get_size(self) -> int:
        """Return the number of processes in a communicator."""
        return self.size

    def Barrier(self) -> None:
        """Barrier synchronization."""
        self.barrier()

    def Allreduce(self, sendbuf, recvbuf, op=mpi_comm.ReduceOperation.SUM) -> None:
        """Reduce to All."""
        if sendbuf is InPlace.IN_PLACE:
            sendbuf = recvbuf
        else:
            raise NotImplementedError("sendbuf with not IN_PLACE")

        if not isinstance(recvbuf, np.ndarray):
            raise NotImplementedError("recvbuf with not np.ndarray")

        if op is not mpi_comm.ReduceOperation.SUM:
            raise NotImplementedError("op with not SUM")

        recvbuf[:] = self.allreduce(sendbuf)

    def Iallreduce(self, sendbuf, recvbuf, op=mpi_comm.ReduceOperation.SUM) -> Request:
        """Nonblocking Reduce to All."""
        # FIXME: Implement async variants
        self.Allreduce(sendbuf, recvbuf, op)
        return Request()


# Exports
IN_PLACE = InPlace.IN_PLACE

SUM = mpi_comm.ReduceOperation.SUM

COMM_WORLD = Intracomm()
