"""Message Passing Interface (client)"""

import os
import enum
import functools
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


class Op(enum.Enum):
    """Reduction operation."""
    SUM = enum.auto()


class Request:
    """Request handler."""

    def wait(self) -> None:
        """Wait for a non-blocking operation to complete."""


class Comm:
    """Communication context."""

    @functools.cached_property
    def size(self) -> int:
        """Communication size"""
        # Lazily initialized to prevent module imports execution
        return int(os.environ["OMPI_COMM_WORLD_SIZE"])

    @functools.cached_property
    def rank(self) -> int:
        """Communication identifier"""
        # Lazily initialized to prevent module imports execution
        return int(os.environ["OMPI_COMM_WORLD_RANK"])

    @functools.cached_property
    def _comm(self) -> comms.Communication:
        """Communication connection"""
        # Lazily initialized to prevent module imports execution
        return comms.Client()

    def _send(self, operation: mpi_comm.Operation, obj) -> None:
        """Send object to server"""
        request = mpi_comm.Request(rank=self.rank, size=self.size, operation=operation, obj=obj)
        self._comm.put(request)

    def _recv_many(self, size=None):
        """Recive objects to server"""
        if size is None:
            size = self.size

        for _ in range(size):
            response: mpi_comm.Response = self._comm.get()
            yield response.obj

    def _recv(self):
        """Recive object to server"""
        return next(self._recv_many(size=1))

    def Disconnect(self) -> None:
        """Disconnect from a communicator."""
        if "_comm" in self.__dict__:
            self._comm.close()

    def __del__(self) -> None:
        """Best effort finalizer"""
        try:
            self.Disconnect()
        except:  # noqa: E722
            pass

    def Get_rank(self) -> int:
        """Return the rank of this process in a communicator."""
        return self.rank

    def Get_size(self) -> int:
        """Return the number of processes in a communicator."""
        return self.size

    def bcast(self, obj, rank=0):
        """Broadcast."""
        if rank == self.rank:
            self._send(operation=mpi_comm.Operation.BROADCAST, obj=obj)
        return self._recv()

    def Barrier(self) -> None:
        """Barrier synchronization."""
        self.allgather(None)

    def allgather(self, obj):
        """Gather to All."""
        self._send(operation=mpi_comm.Operation.GATHER, obj=obj)
        return list(self._recv_many())

    def Allreduce(self, sendbuf, recvbuf, op=Op.SUM) -> None:
        """Reduce to All."""
        if sendbuf is InPlace.IN_PLACE:
            sendbuf = recvbuf
        else:
            raise NotImplementedError("sendbuf with not IN_PLACE")

        if not isinstance(recvbuf, np.ndarray):
            raise NotImplementedError("recvbuf with not np.ndarray")

        if op is not Op.SUM:
            raise NotImplementedError("op with not SUM")

        self._send(operation=mpi_comm.Operation.REDUCE, obj=sendbuf)
        recvbuf[:] = self._recv()

    def Iallreduce(self, sendbuf, recvbuf, op=Op.SUM) -> Request:
        """Nonblocking Reduce to All (fake)"""
        self.Allreduce(sendbuf, recvbuf, op)
        return Request()


# Exports
IN_PLACE = InPlace.IN_PLACE

SUM = Op.SUM

COMM_WORLD = Comm()
