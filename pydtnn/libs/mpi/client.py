"""Message Passing Interface (client)"""

# NOTE: Due to how Python and external libraries handle threading, there is no
# reliable way to track when the MPI context should be automatically finalized.
#
# MPI for Python finalizes its context via a atexit handler, which waits for all
# non-daemon threads to finish before automatically finalizing (if not disabled).
#
# This implementation attempts to finalizes its context specifically when the main
# thread finishs. As other threads might be internal implementation details and
# might be required during the finalization stage. Waiting for a atexit handler
# might lead to those internal threads being reaped, therefor being unable to
# gracefully finalize.
#
# If this approach is not plausible, as it is Python implementation dependant,
# a classic atexit handler is also registered, but as mention before, this
# might lead to an undefined finaliztion behaviour.
#
# To ensure defined gracefull finalization the only reliable method is to manually
# call Finalize, and ensure it is always called, even when exceptions might be
# unhandled and lead to thread termination.

# FIXME: Use semaphore for reponse request counting and threading condition for
# respone recive notifications. Current implementation, when multiple async operations
# are inflight, could issue more recives than expected, leading to a infinite lock.

import uuid
import enum
import typing
import atexit
import functools
import threading
import itertools
from collections import abc
from concurrent.futures import ThreadPoolExecutor, Future

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


class Request[T]:
    """Request handler."""
    def __init__(self, future: Future[T]) -> None:
        self._future = future

    def wait(self) -> T:
        """Wait for a non-blocking operation to complete."""
        return self._future.result()


class Intracomm:
    """Intracommunicator."""

    def __init__(self) -> None:
        """Communicator initialization"""
        self._comm_lock = threading.Lock()

        self._closed = False
        self._respones = dict[uuid.UUID, typing.Any]()

        self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix=f"{__name__}.{self.__class__.__qualname__}:{id(self)}")

    def _recive_response(self) -> None:
        """Recive one response from communication"""
        response = self._comm.get().obj

        match response:
            case mpi_comm.OperationResponse():
                pass
            case _:
                raise RuntimeError(f"Unknown response {response}")

        self._respones[response.id] = response.obj

    def _get_response(self, id: uuid.UUID):
        """Get a particular response (blocking until ready)"""
        while True:
            try:
                return self._respones.pop(id)
            except KeyError:
                self._recive_response()

    def _resolve_request(self, request: mpi_comm.OperationRequest):
        """Resolve request"""
        self._comm.put(request)

        response_id: uuid.UUID = self._get_response(request.id)
        response = self._get_response(response_id)

        if isinstance(response, mpi_comm.RemoteException):
            raise response
        return response

    @property
    def size(self) -> int:
        """Communication size"""
        # NOTE: Lazily initialized, prevent module imports execution
        return mpi_comm.get_size()

    @property
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
                comm = self.__dict__["_comm"] = self._new_comm()
        return comm

    def _new_comm(self) -> comms.Communication:
        """Create a new communication and inizialize it"""
        addr = mpi_comm.get_addr()
        port = mpi_comm.get_port()
        comm = comms.Client(addr=addr, port=port)
        state = mpi_comm.RankInit(rank=self.rank)
        comm.put(state)
        while True:
            response = comm.get().obj

            match response:
                case mpi_comm.StateResponse():
                    pass
                case _:
                    continue
                    # raise RuntimeError(f"Unknown response {response}")

            if response.size == self.size:
                break
        return comm

    def _close_comm(self, comm: comms.Communication) -> None:
        """Fianlize a communication object"""
        state = mpi_comm.RankFinalize()
        comm.put(state)
        while True:
            response = comm.get().obj

            match response:
                case mpi_comm.StateResponse():
                    pass
                case _:
                    continue
                    # raise RuntimeError(f"Unknown response {response}")

            if response.size == 0:
                break
        comm.close()

    def Disconnect(self) -> None:
        """Disconnect from a communicator."""
        # Atomicly get and remove communicator
        if comm := self.__dict__.pop("_comm", None):
            self._close_comm(comm)

        if self._closed:
            return
        self._closed = True

        self._pool.shutdown()

    def __del__(self) -> None:
        """Best effort finalizer"""
        try:
            self.Disconnect()
        except:  # noqa: E722
            pass

    def _submit_operation(self, comm: mpi_comm.CommmunicationGroup, context: mpi_comm.OperationContext, obj: typing.Any) -> Future:
        """Schedule a new operation"""

        # Create appropriate request according to participation
        if self.rank == comm.root:
            request = mpi_comm.OperationRequest(comm=comm, context=context, obj=obj)
        elif self.rank in comm.src:
            request = mpi_comm.OperationRequest(comm=comm, obj=obj)
        elif self.rank in comm.dst:
            request = mpi_comm.OperationRequest(comm=comm)
        else:
            raise RuntimeError("Tried to schedule operation without participating")

        return self._pool.submit(self._resolve_request, request)

    def bcast[T](self, obj: T, root: mpi_comm.Rank = 0) -> T:
        """Broadcast."""
        context = mpi_comm.BroadcastContext(root=root)
        comm = context.comm(size=self.size)
        return self._submit_operation(comm=comm, obj=obj, context=context).result()

    def barrier(self) -> None:
        """Barrier synchronization."""
        self.allreduce(0)

    def allgather[T](self, obj: T) -> list[T]:
        """Gather to All."""
        context = mpi_comm.AllGatherContext()
        comm = context.comm(size=self.size)
        return self._submit_operation(comm=comm, obj=obj, context=context).result()

    def allreduce[T](self, obj: T, op: mpi_comm.ReduceOperation = mpi_comm.ReduceOperation.SUM) -> T:
        """Reduce to All."""
        if op is not mpi_comm.ReduceOperation.SUM:
            raise NotImplementedError("op with not SUM")

        context = mpi_comm.AllReduceContext(op=op)
        comm = context.comm(size=self.size)
        return self._submit_operation(comm=comm, obj=obj, context=context).result()

    def _phased_allreduce[T](self, obj: T, op: mpi_comm.ReduceOperation = mpi_comm.ReduceOperation.SUM) -> T:
        """Reduce to All (with steps)."""
        if op is not mpi_comm.ReduceOperation.SUM:
            raise NotImplementedError("op with not SUM")

        context = mpi_comm.AllPhasedReduceContext(op=op)

        for phase in itertools.count():
            comm = context.comm(rank=self.rank, size=self.size, phase=phase)
            obj = self._submit_operation(comm=comm, obj=obj, context=context).result()
            if len(comm.dst) == self.size:
                break

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

    def Allreduce[T: abc.Buffer](self, sendbuf: T | typing.Literal[InPlace.IN_PLACE], recvbuf: T, op: mpi_comm.ReduceOperation = mpi_comm.ReduceOperation.SUM) -> None:
        """Reduce to All."""
        self.Iallreduce(sendbuf, recvbuf, op).wait()

    def Iallreduce[T: abc.Buffer](self, sendbuf: T | typing.Literal[InPlace.IN_PLACE], recvbuf: T, op: mpi_comm.ReduceOperation = mpi_comm.ReduceOperation.SUM) -> Request[T]:
        """Nonblocking Reduce to All."""
        if sendbuf is InPlace.IN_PLACE:
            sendbuf = recvbuf

        if not isinstance(recvbuf, np.ndarray):
            raise NotImplementedError("recvbuf with not np.ndarray")

        if op is not mpi_comm.ReduceOperation.SUM:
            raise NotImplementedError("op with not SUM")

        def callback(future: Future[T]):
            recvbuf[:] = future.result()

        context = mpi_comm.AllReduceContext(op=op)
        comm = context.comm(size=self.size)
        future = self._submit_operation(comm=comm, obj=sendbuf, context=context)
        future.add_done_callback(callback)
        return Request(future)


# Exports
IN_PLACE = InPlace.IN_PLACE

SUM = mpi_comm.ReduceOperation.SUM

COMM_WORLD = Intracomm()

# Best effort finalizer
try:
    threading._register_atexit(Finalize)  # type: ignore (private implementation dependant)
except AttributeError:
    atexit.register(Finalize)
