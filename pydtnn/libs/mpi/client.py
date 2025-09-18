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

# TODO: Move request.callback to thread_queue to avoid callback costs

# TODO: Revise gather v-variants

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
from concurrent.futures import Future

from pydtnn import comms, utils
from pydtnn.utils import asynctools
from pydtnn.utils.io_stream import byteview
from pydtnn.libs.mpi import comm as mpi_comm


__all__ = (
    "Finalize",
    "IN_PLACE",
    "MAX",
    "MIN",
    "SUM",
    "PROD",
    "LAND",
    "BAND",
    "LOR",
    "BOR",
    "LXOR",
    "BXOR",
    "MINLOC",
    "MAXLOC",
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
    RES = enum.auto()
    FIN = enum.auto()


class Request[T]:
    """Request handler."""

    def __init__(self) -> None:
        """Inizialize request"""
        self._state = RequestState.INI
        self._lock = threading.Lock()
        self._future = Future[T]()

    @staticmethod
    def _process[O](result: O) -> O:
        """Result post-processing"""
        if isinstance(result, mpi_comm.RemoteException):
            raise result
        return result

    def _resolve(self, future: Future[T]) -> None:
        """Resolve request according to future"""
        exc = future.exception()
        self._state = RequestState.FIN

        if exc is None:
            result = future.result()
            asynctools.future_set_result(self._future, result)
        else:
            asynctools.future_set_exception(self._future, exc)

    def wait(self) -> T:
        """Wait for a non-blocking operation to complete."""
        with self._lock:
            if "_future" in self.__dict__:
                result = self._future.result()
                del self._future
            else:
                result = None
        return result  # type: ignore


class Comm:
    """Communicator."""

    def __init__(self) -> None:
        """Communicator initialization"""
        self._comm_lock = threading.Lock()

        self._close_init = threading.Lock()
        self._close_done = threading.Event()

        self._requests = dict[uuid.UUID, Request]()
        self._responses = dict[uuid.UUID, typing.Any]()

        thread_prefix = f"{__name__}.{self.__class__.__qualname__}:{id(self)}"
        self._comm_queue = utils.thread_queue(f"{thread_prefix}.comm")
        self._task_queue = utils.thread_queue(f"{thread_prefix}.task")

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
        while id in self._requests and id in self._responses:
            request = self._requests.pop(id)
            response = self._responses.pop(id)

            match request._state:
                case RequestState.INI:
                    id = response
                    self._requests[id] = request
                    request._state = RequestState.ACK
                case RequestState.ACK:
                    request._state = RequestState.RES
                    if isinstance(response, mpi_comm.RemoteException):
                        request._process = Request._process  # Remove callback
                    future = self._task_queue.submit(request._process, response)
                    future.add_done_callback(request._resolve)
                case _:
                    raise RuntimeError(f"Invalid request state {request._state}")

    def _shedule_operation(self, comm: mpi_comm.CommmunicationGroup, context: mpi_comm.OperationContext, obj: typing.Any, process: abc.Callable = Request._process) -> Request:
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
        request._process = process
        asynctools.future_set_running(request._future)

        if self.rank in comm.dst:
            self._requests[operation.id] = request

        future = self._comm.put(operation)

        if self.rank in comm.dst:
            self._comm_queue.submit(self._recive_response).add_done_callback(lambda future: future.result())
            self._comm_queue.submit(self._recive_response).add_done_callback(lambda future: future.result())
        else:
            future.add_done_callback(request._resolve)

        return request

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
        comm_options = comms.CommunicatorOptions(netloc=comms.NetworkLocation(host=addr, port=port))
        state = mpi_comm.RankInit(rank=self.rank)
        try:
            comm = comms.Client(comm_options)
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
        with self._comm_lock:
            if "_comm" in self.__dict__:
                self.barrier()

            self._comm_queue.shutdown()
            self._task_queue.shutdown()

            if comm := self.__dict__.pop("_comm", None):
                self._close_comm(comm)

    def Disconnect(self) -> None:
        """Disconnect from a communicator."""
        if self._close_init.acquire(blocking=False):
            self._close()
            self._close_done.set()
        self._close_done.wait()

    def __del__(self) -> None:
        """Best effort finalizer"""
        try:
            self.Disconnect()
        except:  # noqa: E722
            pass

    # Properties
    @property
    def size(self) -> int:
        """Communication size"""
        return mpi_comm.get_size()

    @property
    def rank(self) -> mpi_comm.Rank:
        """Communication identifier"""
        return mpi_comm.get_rank()

    def Get_rank(self) -> int:
        """Return the rank of this process in a communicator."""
        return self.rank

    def Get_size(self) -> int:
        """Return the number of processes in a communicator."""
        return self.size

    # Barrier synchronization
    def ibarrier(self) -> Request[None]:
        """Nonblocking Barrier synchronization."""
        return self.iallreduce(None, mpi_comm.ReduceOperation.LAND)

    def barrier(self) -> None:
        """Barrier synchronization."""
        return self.ibarrier().wait()

    def Ibarrier(self) -> Request[None]:
        """Nonblocking Barrier synchronization."""
        return self.ibarrier()

    def Barrier(self) -> None:
        """Barrier synchronization."""
        return self.Ibarrier().wait()

    # Broadcast
    def ibcast[T](self, obj: T, root: mpi_comm.Rank = 0) -> Request[T]:
        """Broadcast."""
        context = mpi_comm.BroadcastContext(root=root)
        comm = context.comm(size=self.size)
        return self._shedule_operation(comm=comm, context=context, obj=obj)

    def bcast[T](self, obj: T, root: mpi_comm.Rank = 0) -> T:
        """Broadcast."""
        return self.ibcast(obj=obj, root=root).wait()

    def Ibcast[T: abc.Buffer](self, buf: T, root: mpi_comm.Rank = 0) -> Request[None]:
        """Nonblocking Gather to All."""
        context = mpi_comm.BroadcastContext(root=root)
        comm = context.comm(size=self.size)

        def process(result: T) -> None:
            with byteview(result) as src, byteview(buf) as dst:
                dst[:] = src

        return self._shedule_operation(comm=comm, context=context, obj=buf, process=process)

    def Bcast[T: abc.Buffer](self, buf: T, root: mpi_comm.Rank = 0) -> None:
        """Gather to All."""
        return self.Ibcast(buf=buf, root=root).wait()

    # Gather to All
    def iallgather[T](self, sendobj: T) -> Request[list[T]]:
        """Nonblocking Gather to All."""
        context = mpi_comm.AllGatherContext()
        comm = context.comm(size=self.size)
        return self._shedule_operation(comm=comm, context=context, obj=sendobj)

    def allgather[T](self, sendobj: T) -> list[T]:
        """Gather to All."""
        return self.iallgather(sendobj=sendobj).wait()

    def Iallgather[T: abc.Buffer](self, sendbuf: T | typing.Literal[InPlace.IN_PLACE], recvbuf: T) -> Request[None]:
        """Nonblocking Gather to All."""
        if sendbuf is InPlace.IN_PLACE:
            sendbuf = recvbuf

        def process(results: list[T]) -> None:
            offset = 0
            with byteview(recvbuf) as dst:
                for result in results:
                    with byteview(result) as src:
                        dst[offset:offset + len(src)] = src
                        offset += len(src)

        context = mpi_comm.AllGatherContext()
        comm = context.comm(size=self.size)
        return self._shedule_operation(comm=comm, context=context, obj=sendbuf, process=process)

    def Allgather[T: abc.Buffer](self, sendbuf: T | typing.Literal[InPlace.IN_PLACE], recvbuf: T) -> None:
        """Gather to All."""
        self.Iallgather(sendbuf=sendbuf, recvbuf=recvbuf).wait()

    # Reduce to All
    def iallreduce[T](self, sendobj: T, op: mpi_comm.ReduceOperation = mpi_comm.ReduceOperation.SUM) -> Request[T]:
        """Reduce to All."""
        context = mpi_comm.AllReduceContext(op=op)
        comm = context.comm(size=self.size)
        return self._shedule_operation(comm=comm, context=context, obj=sendobj)

    def allreduce[T](self, sendobj: T, op: mpi_comm.ReduceOperation = mpi_comm.ReduceOperation.SUM) -> T:
        """Reduce to All."""
        return self.iallreduce(sendobj=sendobj, op=op).wait()

    def Iallreduce[T: abc.Buffer](self, sendbuf: T | typing.Literal[InPlace.IN_PLACE], recvbuf: T, op: mpi_comm.ReduceOperation = mpi_comm.ReduceOperation.SUM) -> Request[None]:
        """Nonblocking Reduce to All."""
        if sendbuf is InPlace.IN_PLACE:
            sendbuf = recvbuf

        def process(result: T) -> None:
            with byteview(result) as src, byteview(recvbuf) as dst:
                dst[:] = src

        context = mpi_comm.AllReduceContext(op=op)
        comm = context.comm(size=self.size)
        return self._shedule_operation(comm=comm, context=context, obj=sendbuf, process=process)

    def Allreduce[T: abc.Buffer](self, sendbuf: T | typing.Literal[InPlace.IN_PLACE], recvbuf: T, op: mpi_comm.ReduceOperation = mpi_comm.ReduceOperation.SUM) -> None:
        """Reduce to All."""
        self.Iallreduce(sendbuf=sendbuf, recvbuf=recvbuf, op=op).wait()

    # Reduce to All (with steps)
    def _phased_allreduce[T](self, sendobj: T, op: mpi_comm.ReduceOperation = mpi_comm.ReduceOperation.SUM) -> T:
        """Reduce to All (with steps)."""
        context = mpi_comm.AllPhasedReduceContext(op=op)

        for phase in itertools.count():
            comm = context.comm(rank=self.rank, size=self.size, phase=phase)
            sendobj = self._shedule_operation(comm=comm, context=context, obj=sendobj).wait()
            if len(comm.dst) == self.size:
                break

        return sendobj

    # Scatter
    def iscatter[T](self, sendobj: abc.Sequence[T], root: mpi_comm.Rank = 0) -> Request[T]:
        """Nonblocking Scatter."""
        context = mpi_comm.ScatterContext(root=root)
        comm = context.comm(size=self.size)
        return self._shedule_operation(comm=comm, context=context, obj=sendobj)

    def scatter[T](self, sendobj: abc.Sequence[T], root: mpi_comm.Rank = 0) -> T:
        """Scatter."""
        return self.iscatter(sendobj=sendobj, root=root).wait()

    # All to All Scatter/Gather
    def ialltoall[T](self, sendobj: abc.Sequence[T]) -> Request[list[T]]:
        """All to All Scatter/Gather."""
        context = mpi_comm.AllToAllContext()
        comm = context.comm(size=self.size)
        return self._shedule_operation(comm=comm, context=context, obj=sendobj)

    def alltoall[T](self, sendobj: abc.Sequence[T]) -> list[T]:
        """All to All Scatter/Gather."""
        return self.ialltoall(sendobj=sendobj).wait()

    # Gather to All
    def igather[T](self, sendobj: T, root: mpi_comm.Rank = 0) -> Request[list[T]]:
        """Nonblocking Gather."""
        context = mpi_comm.GatherContext(root=root)
        comm = context.comm(size=self.size)
        return self._shedule_operation(comm=comm, context=context, obj=sendobj)

    def gather[T](self, sendobj: T, root: mpi_comm.Rank = 0) -> list[T]:
        """Gather."""
        return self.igather(sendobj=sendobj, root=root).wait()

    def Igather[T: abc.Buffer](self, sendbuf: T | typing.Literal[InPlace.IN_PLACE], recvbuf: T, root: mpi_comm.Rank = 0) -> Request[None]:
        """Nonblocking Gather."""
        if sendbuf is InPlace.IN_PLACE:
            sendbuf = recvbuf

        def process(results: list[T]) -> None:
            offset = 0
            with byteview(recvbuf) as dst:
                for result in results:
                    with byteview(result) as src:
                        dst[offset:offset + len(src)] = src
                        offset += len(src)

        context = mpi_comm.GatherContext(root=root)
        comm = context.comm(size=self.size)
        return self._shedule_operation(comm=comm, context=context, obj=sendbuf, process=process)

    def Gather[T: abc.Buffer](self, sendbuf: T | typing.Literal[InPlace.IN_PLACE], recvbuf: T, root: mpi_comm.Rank = 0) -> None:
        """Gather."""
        self.Igather(sendbuf=sendbuf, recvbuf=recvbuf, root=root).wait()

    # Reduce
    def ireduce[T](self, sendobj: T, op: mpi_comm.ReduceOperation = mpi_comm.ReduceOperation.SUM, root: mpi_comm.Rank = 0) -> Request[T]:
        """Reduce to All."""
        context = mpi_comm.ReduceContext(op=op, root=root)
        comm = context.comm(size=self.size)
        return self._shedule_operation(comm=comm, context=context, obj=sendobj)

    def reduce[T](self, sendobj: T, op: mpi_comm.ReduceOperation = mpi_comm.ReduceOperation.SUM, root: mpi_comm.Rank = 0) -> T:
        """Reduce to All."""
        return self.ireduce(sendobj=sendobj, op=op, root=root).wait()

    def Ireduce[T: abc.Buffer](self, sendbuf: T | typing.Literal[InPlace.IN_PLACE], recvbuf: T, op: mpi_comm.ReduceOperation = mpi_comm.ReduceOperation.SUM, root: mpi_comm.Rank = 0) -> Request[None]:
        """Nonblocking Reduce to All."""
        if sendbuf is InPlace.IN_PLACE:
            sendbuf = recvbuf

        def process(result: T) -> None:
            with byteview(result) as src, byteview(recvbuf) as dst:
                dst[:] = src

        context = mpi_comm.ReduceContext(op=op, root=root)
        comm = context.comm(size=self.size)
        return self._shedule_operation(comm=comm, context=context, obj=sendbuf, process=process)

    def Reduce[T: abc.Buffer](self, sendbuf: T | typing.Literal[InPlace.IN_PLACE], recvbuf: T, op: mpi_comm.ReduceOperation = mpi_comm.ReduceOperation.SUM, root: mpi_comm.Rank = 0) -> None:
        """Reduce to All."""
        self.Ireduce(sendbuf=sendbuf, recvbuf=recvbuf, op=op, root=root).wait()


# Exports
IN_PLACE = InPlace.IN_PLACE

MAX = mpi_comm.ReduceOperation.MAX
MIN = mpi_comm.ReduceOperation.MIN
SUM = mpi_comm.ReduceOperation.SUM
PROD = mpi_comm.ReduceOperation.PROD
LAND = mpi_comm.ReduceOperation.LAND
BAND = mpi_comm.ReduceOperation.BAND
LOR = mpi_comm.ReduceOperation.LOR
BOR = mpi_comm.ReduceOperation.BOR
LXOR = mpi_comm.ReduceOperation.LXOR
BXOR = mpi_comm.ReduceOperation.BXOR
MINLOC = mpi_comm.ReduceOperation.MINLOC
MAXLOC = mpi_comm.ReduceOperation.MAXLOC

COMM_WORLD = Comm()

# Best effort finalizer
try:
    threading._register_atexit(Finalize)  # type: ignore (private implementation dependant)
except AttributeError:
    atexit.register(Finalize)
