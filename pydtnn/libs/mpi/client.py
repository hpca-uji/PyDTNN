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

# FIXME: Move backgroud_server logic from communicator to module. Currently it is
# here because of import restrictions, but it will be an issue when multiple
# communicator are open.

import uuid
import enum
import typing
import atexit
import warnings
import functools
import threading
import itertools
from collections import abc
from concurrent.futures import ThreadPoolExecutor

from pydtnn import comms
from pydtnn.utils.io_stream import byteview
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


class RequestState(enum.Enum):
    INI = enum.auto()
    ACK = enum.auto()
    FIN = enum.auto()


class Request[T]:
    """Request handler."""
    _result: T
    _callback: abc.Callable[[T], T]

    def __init__(self) -> None:
        """Inizialize request"""
        self._state = RequestState.INI
        self._lock = threading.Condition()

    def _put(self, value) -> uuid.UUID | None:
        """Process state change"""
        match self._state:
            case RequestState.INI:
                self._state = RequestState.ACK
                return value

            case RequestState.ACK:
                self._result = value
                with self._lock:
                    self._state = RequestState.FIN
                    self._lock.notify_all()
                return None

            case _:
                raise RuntimeError(f"Invalid request state {self._state}")

    def wait(self) -> T:
        """Wait for a non-blocking operation to complete."""
        with self._lock:
            self._lock.wait_for(lambda: self._state == RequestState.FIN)
        result = self.__dict__.pop("_result", None)
        if isinstance(result, mpi_comm.RemoteException):
            raise result
        callback = self.__dict__.pop("_callback", lambda result: result)
        return callback(result)


class Comm:
    """Communicator."""

    def __init__(self) -> None:
        """Communicator initialization"""
        self._close_lock = threading.Lock()
        self._comm_lock = threading.Lock()

        self._requests = dict[uuid.UUID, Request]()
        self._responses = dict[uuid.UUID, typing.Any]()
        self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix=f"{__name__}.{self.__class__.__qualname__}:{id(self)}")

    def _recive_response(self) -> None:
        """Recive one response from communication"""
        response = self._comm.get().obj

        match response:
            case mpi_comm.OperationResponse():
                pass
            case _:
                raise RuntimeError(f"Unknown response {response}")

        self._responses[response.id] = response.obj

        # Process pending requests
        self._handle_request(response.id)

    def _handle_request(self, id: uuid.UUID) -> None:
        """Handle a request"""
        # While matching request and response
        while request := self._requests.pop(id, None):
            if (response := self._responses.pop(id, self)) is self:
                self._requests[id] = request
                break

            # Continue request
            if id := request._put(response):  # type: ignore
                self._requests[id] = request

            # Finish request
            else:
                break

    def _submit_operation(self, comm: mpi_comm.CommmunicationGroup, context: mpi_comm.OperationContext, obj: typing.Any) -> Request:
        """Schedule a new operation"""

        # Create appropriate request according to participation
        if self.rank == comm.root:
            operation = mpi_comm.OperationRequest(comm=comm, context=context, obj=obj)
        elif self.rank in comm.src:
            operation = mpi_comm.OperationRequest(comm=comm, obj=obj)
        elif self.rank in comm.dst:
            operation = mpi_comm.OperationRequest(comm=comm)
        else:
            raise RuntimeError("Tried to schedule operation without participating")

        request = Request()
        self._requests[operation.id] = request
        self._comm.put(operation)
        if self.rank in comm.dst:
            self._pool.submit(self._recive_response).add_done_callback(lambda future: future.result())
            self._pool.submit(self._recive_response).add_done_callback(lambda future: future.result())
        return request

    @property
    def size(self) -> int:
        """Communication size"""
        return mpi_comm.get_size()

    @property
    def rank(self) -> mpi_comm.Rank:
        """Communication identifier"""
        return mpi_comm.get_rank()

    @functools.cached_property
    def _comm(self) -> comms.Communicator:
        """Communication connection"""
        # NOTE: Lazily initialized, prevent module imports execution
        with self._comm_lock:
            if "_comm" in self.__dict__:
                comm = self.__dict__["_comm"]
            else:
                comm = self.__dict__["_comm"] = self._new_comm()
        return comm

    def _new_comm(self) -> comms.Communicator:
        """Create a new communication and inizialize it"""

        # If requested, start a local server
        if mpi_comm.get_init():
            if self.rank == 0:
                from pydtnn.libs.mpi.server import background_server
                self._server = background_server()

            # Allow some time for server startup
            from time import sleep
            sleep(0.5)

        addr = mpi_comm.get_addr()
        port = mpi_comm.get_port()
        state = mpi_comm.RankInit(rank=self.rank)
        try:
            comm = comms.Client({"addr": addr, "port": port})
            comm.put(state)
            while True:
                response = comm.get().obj

                match response:
                    case mpi_comm.StateResponse():
                        pass
                    case _:
                        warnings.warn(f"Unknown response {response}", RuntimeWarning)
                        continue  # response lost

                if response.size == self.size:
                    break
        except Exception:
            try:
                comm.close()
            except:  # noqa: E722
                pass
            raise
        return comm

    def _close_comm(self, comm: comms.Communicator) -> None:
        """Fianlize a communication object"""
        state = mpi_comm.RankFinalize()
        try:
            comm.put(state)
            while True:
                response = comm.get().obj

                match response:
                    case mpi_comm.StateResponse():
                        pass
                    case _:
                        warnings.warn(f"Unknown response {response}", RuntimeWarning)
                        continue  # response lost

                if response.size == 0:
                    break
        finally:
            comm.close()

        # If requested, stop a local server
        if mpi_comm.get_init():
            if self.rank == 0:
                self._server.result()

            # Allow some time for server shutdown
            from time import sleep
            sleep(0.5)

    def _close(self) -> None:
        """Communicator finalizer"""
        # Close comunicator if initialized
        if comm := self.__dict__.pop("_comm", None):
            self._close_comm(comm)

        self._pool.shutdown()

    def Disconnect(self) -> None:
        """Disconnect from a communicator."""
        if self._close_lock.acquire(blocking=False):
            self._close()

    def __del__(self) -> None:
        """Best effort finalizer"""
        try:
            self.Disconnect()
        except:  # noqa: E722
            pass

    def bcast[T](self, obj: T, root: mpi_comm.Rank = 0) -> T:
        """Broadcast."""
        context = mpi_comm.BroadcastContext(root=root)
        comm = context.comm(size=self.size)
        return self._submit_operation(comm=comm, obj=obj, context=context).wait()

    def barrier(self) -> None:
        """Barrier synchronization."""
        self.allreduce(0)

    def allgather[T](self, obj: T) -> list[T]:
        """Gather to All."""
        context = mpi_comm.AllGatherContext()
        comm = context.comm(size=self.size)
        return self._submit_operation(comm=comm, obj=obj, context=context).wait()

    def allreduce[T](self, obj: T, op: mpi_comm.ReduceOperation = mpi_comm.ReduceOperation.SUM) -> T:
        """Reduce to All."""
        return self.iallreduce(obj=obj, op=op).wait()

    def iallreduce[T](self, obj: T, op: mpi_comm.ReduceOperation = mpi_comm.ReduceOperation.SUM) -> Request[T]:
        """Reduce to All."""
        context = mpi_comm.AllReduceContext(op=op)
        comm = context.comm(size=self.size)
        return self._submit_operation(comm=comm, obj=obj, context=context)

    def _phased_allreduce[T](self, obj: T, op: mpi_comm.ReduceOperation = mpi_comm.ReduceOperation.SUM) -> T:
        """Reduce to All (with steps)."""
        context = mpi_comm.AllPhasedReduceContext(op=op)

        for phase in itertools.count():
            comm = context.comm(rank=self.rank, size=self.size, phase=phase)
            obj = self._submit_operation(comm=comm, obj=obj, context=context).wait()
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

        def callback(result: T):
            with byteview(result) as src, byteview(recvbuf) as dst:
                dst[:] = src

        req = self.iallreduce(obj=sendbuf, op=op)
        req._callback = callback  # type: ignore
        return req


# Exports
IN_PLACE = InPlace.IN_PLACE

SUM = mpi_comm.ReduceOperation.SUM

COMM_WORLD = Comm()

# Best effort finalizer
try:
    threading._register_atexit(Finalize)  # type: ignore (private implementation dependant)
except AttributeError:
    atexit.register(Finalize)
