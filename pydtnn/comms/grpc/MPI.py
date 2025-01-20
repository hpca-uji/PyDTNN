"""Message Passing Interface."""

import os
import enum
import pickle
import grpc
import numpy as np
from pydtnn.comms.grpc import mpi_pb2
from pydtnn.comms.grpc import mpi_pb2_grpc

__all__ = (
    "Finalize",
    "IN_PLACE",
    "SUM",
    "COMM_WORLD",
)


StreamResponse = list[mpi_pb2.RecvResponse]


def Finalize():
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
    _pickle_protocol = 5

    def __init__(self):
        """Inizialize comunication context"""
        self.size = int(os.environ["OMPI_COMM_WORLD_SIZE"])
        self.rank = int(os.environ["OMPI_COMM_WORLD_RANK"])
        self._grpc_init()

    def _grpc_init(self) -> None:
        """Inizialize gRPC context"""

        # Setup environment
        self._grpc_host = os.environ.get("GRPC_HOST", "localhost:50051")

        # Client inizialization
        self._grpc = grpc.insecure_channel(self._grpc_host)
        self._grpc_stub = mpi_pb2_grpc.MPIStub(self._grpc)

    def _grpc_finalize(self):
        """Finalize gRPC context"""
        self._grpc.close()

    def _serialize(self, obj) -> bytes:
        """Serialize object for comunication"""
        return pickle.dumps(obj, protocol=self._pickle_protocol)

    def _deserialize(self, data: bytes):
        """Deserialize object for comunication"""
        return pickle.loads(data)

    def _send(self, obj):
        """Send object to server"""
        req = mpi_pb2.SendRequest(rank=self.rank, data=self._serialize(obj))
        self._grpc_stub.send(req)

    def _recv_many(self, op: mpi_pb2.Op):
        """Recive objects to server"""
        req = mpi_pb2.RecvRequest(rank=self.rank, size=self.size, op=op)
        while True:
            try:
                res_steam: StreamResponse = self._grpc_stub.recv(req)
                yield from (self._deserialize(res.data) for res in res_steam)
            except grpc.RpcError as exc:
                if exc.code() is grpc.StatusCode.UNAVAILABLE:  # type: ignore
                    pass
                else:
                    raise
            else:
                break

    def _recv(self, op: mpi_pb2.Op):
        """Recive object to server"""
        return next(self._recv_many(op))

    def Disconnect(self) -> None:
        """Disconnect from a communicator."""
        self._grpc_finalize()

    def __del__(self):
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
            self._send(obj)
        return self._recv(op=mpi_pb2.BCAST)

    def Barrier(self) -> None:
        """Barrier synchronization."""
        self.allgather(None)

    def allgather(self, obj):
        """Gather to All."""
        self._send(obj)
        return list(self._recv_many(op=mpi_pb2.ALLGATHER))

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

        self._send(sendbuf)
        recvbuf[:] = self._recv(op=mpi_pb2.ALLREDUCE)

    def Iallreduce(self, sendbuf, recvbuf, op=Op.SUM):
        """Nonblocking Reduce to All."""
        self.Allreduce(sendbuf, recvbuf, op)
        return Request()


# Exports
IN_PLACE = InPlace.IN_PLACE

SUM = Op.SUM

COMM_WORLD = Comm()
