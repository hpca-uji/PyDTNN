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
import itertools
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
    COMM_WORLD.Disconnect()


class InPlace(enum.Enum):
    """In-place buffer argument."""
    IN_PLACE = enum.auto()


class Request:
    """Request handler."""

    def wait(self) -> None:
        """Wait for a non-blocking operation to complete."""


class Intracomm:
    """Intracommunicator."""

    def __init__(self) -> None:
        """Communicator initialization"""
        self._closed = False
        self._comm_lock = threading.Lock()
        self._state_lock = threading.Lock()
        self._state = dict[mpi_comm.RankGroup, SimpleQueue[mpi_comm.OperationResponse | None]]()

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

    @functools.cached_property
    def _comm(self) -> comms.Communication:
        """Communication connection"""
        # NOTE: Lazily initialized, prevent module imports execution
        with self._comm_lock:
            if comm := self.__dict__.get("_comm"):
                pass
            else:
                self._comm = comm = self._new_comm()
        return comm

    def _comm_get(self):
        """Get one response from communication"""
        assert self._state_lock.locked(), "Modifing state without lock"
        response = self._comm.get().obj

        match response:
            case mpi_comm.OperationResponse():
                pass
            case _:
                raise RuntimeError(f"Unknown response {response}")

        queue = self._state.setdefault(response.dst, SimpleQueue())
        queue.put(response)

    def _new_comm(self) -> comms.Communication:
        """Create a new communication and inizialize it"""
        addr = mpi_comm.get_addr()
        port = mpi_comm.get_port()
        comm = comms.Client(addr=addr, port=port)
        request = mpi_comm.InitRequest(rank=self.rank, size=self.size)
        comm.put(request)
        comm.get()
        return comm

    def _close_comm(self, comm: comms.Communication) -> None:
        """Fianlize a communication object"""
        request = mpi_comm.FinalizeRequest()
        comm.put(request)
        comm.get()
        comm.close()

    def Disconnect(self) -> None:
        """Disconnect from a communicator."""
        # Atomicly get and remove communicator
        if comm := self.__dict__.pop("_comm", None):
            self._close_comm(comm)

        if self._closed:
            return
        self._closed = True

        with self._state_lock:
            for queue in self._state.values():
                queue.put(None)

    def _put(self, request: mpi_comm.OperationRequest) -> None:
        """Publish object to server"""
        if self.rank in request.comm.src:
            self._comm.put(request)

    def _get(self, request: mpi_comm.OperationRequest):
        """Get object from server"""
        group = request.comm.dst

        with self._state_lock:
            queue = self._state.setdefault(group, SimpleQueue())

            while True:
                try:
                    response = queue.get_nowait()
                except Empty:
                    self._comm_get()
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
        request = mpi_comm.BroadcastRequest(rank=self.rank, size=self.size, obj=obj, root=root)
        self._put(request)
        return self._get(request)

    def barrier(self) -> None:
        """Barrier synchronization."""
        self.allreduce(0)

    def allgather(self, obj):
        """Gather to All."""
        request = mpi_comm.AllGatherRequest(rank=self.rank, size=self.size, obj=obj)
        self._put(request)
        return list(self._get(request) for _ in range(self.size))

    def allreduce(self, obj, op: mpi_comm.ReduceOperation = mpi_comm.ReduceOperation.SUM):
        """Reduce to All."""
        if op is not mpi_comm.ReduceOperation.SUM:
            raise NotImplementedError("op with not SUM")

        request = mpi_comm.AllReduceRequest(rank=self.rank, size=self.size, obj=obj, op=op)
        self._put(request)
        return self._get(request)

    def _phased_allreduce(self, obj, op: mpi_comm.ReduceOperation = mpi_comm.ReduceOperation.SUM):
        """Reduce to All (with steps)."""
        if op is not mpi_comm.ReduceOperation.SUM:
            raise NotImplementedError("op with not SUM")

        for phase in itertools.count():
            request = mpi_comm.AllPhasedReduceRequest(rank=self.rank, size=self.size, obj=obj, op=op, phase=phase)
            self._put(request)
            obj = self._get(request)
            if len(request.comm.dst) == self.size:
                return obj

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
        # FIXME: Implement async communications
        self.Allreduce(sendbuf, recvbuf, op)
        return Request()


# Exports
IN_PLACE = InPlace.IN_PLACE

SUM = mpi_comm.ReduceOperation.SUM

COMM_WORLD = Intracomm()

atexit.register(Finalize)
