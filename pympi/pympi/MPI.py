"""Message Passing Interface (client)"""

# TODO: Collective: Revise vector variants

# TODO: P2P: self-messaging, tagging, any source

# TODO: Move backgroud_server logic from communicator to module. Currently it is
# here because of import restrictions, but it will be an issue when multiple
# communicator are open.

import copy
import uuid
import enum
import typing
import atexit
import warnings
import functools
import threading
import itertools
from collections import abc
from concurrent import futures
from concurrent.futures import Future

import net_queue as nq
from net_queue import asynctools
from pympi import proto, rc, util
from net_queue.io_stream import byteview
from net_queue.asynctools import thread_queue


__all__ = (
    "Init",
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


ANY_TAG: proto.Tag = 0
ANY_SOURCE: proto.Rank = -1


def Init() -> None:
    """Initialize the MPI execution environment."""


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
        self.future = Future[T]()

    @staticmethod
    def _callback[O](result: O) -> O:
        """Result post-processing"""
        if isinstance(result, proto.RemoteException):
            raise result
        return result

    def _resolve(self, future: Future[T]) -> None:
        """Resolve request according to future"""
        exc = future.exception()
        self._state = RequestState.FIN

        if exc is None:
            result = future.result()
            asynctools.future_set_result(self.future, result)
        else:
            asynctools.future_set_exception(self.future, exc)

    def wait(self, status=None) -> T:
        """Wait for a non-blocking operation to complete."""
        if status:
            raise ValueError("Status are not supported")

        with self._lock:
            if "future" in self.__dict__:
                result = self.future.result()
                del self.future
            else:
                result = None

        return result  # type: ignore

    @classmethod
    def Waitall(cls, requests: abc.Sequence["Request[T]"], statuses=None) -> list[int]:
        """Wait for all previously initiated requests to complete"""
        if statuses:
            raise ValueError("Status are not supported")

        fs = [request.future for request in requests]
        done = futures.wait(fs=fs, return_when=futures.ALL_COMPLETED).done

        return list(map(fs.index, done))

    @classmethod
    def Waitsome(cls, requests: abc.Sequence["Request[T]"], statuses=None) -> list[int]:
        """Wait for some previously initiated requests to complete"""
        if statuses:
            raise ValueError("Status are not supported")

        fs = [request.future for request in requests]
        done = futures.wait(fs=fs, return_when=futures.FIRST_COMPLETED).done

        return list(map(fs.index, done))

    @classmethod
    def Waitany(cls, requests: abc.Sequence["Request[T]"], statuses=None) -> int:
        """Wait for any previously initiated request to complete"""
        if statuses:
            raise ValueError("Status are not supported")

        return cls.Waitsome(requests, statuses)[0]

    @classmethod
    def waitall(cls, requests: abc.Sequence["Request[T]"], statuses=None) -> list[T]:
        """Wait for all previously initiated requests to complete"""
        return [requests[i].wait() for i in cls.Waitall(requests, statuses)]

    @classmethod
    def waitsome(cls, requests: abc.Sequence["Request[T]"], statuses=None) -> list[T]:
        """Wait for some previously initiated requests to complete"""
        return [requests[i].wait() for i in cls.Waitsome(requests, statuses)]

    @classmethod
    def waitany(cls, requests: abc.Sequence["Request[T]"], statuses=None) -> T:
        """Wait for any previously initiated request to complete"""
        return requests[cls.Waitany(requests, statuses)].wait()


class Comm:
    """Communicator."""

    def __init__(self, comm_options: nq.CommunicatorOptions = nq.CommunicatorOptions()) -> None:
        """Communicator initialization"""
        self._comm_options = copy.replace(util.comm_options(comm_options))

        self._comm_lock = threading.Lock()

        self._close_init = threading.Lock()
        self._close_done = threading.Event()

        self._requests = dict[uuid.UUID, Request]()
        self._responses = dict[uuid.UUID, typing.Any]()

        thread_prefix = f"{__name__}.{self.__class__.__qualname__}:{id(self)}"
        self._recv_queue = thread_queue(f"{thread_prefix}.recv")
        self._resolve_queue = thread_queue(f"{thread_prefix}.resolve")

    @property
    def _closed(self):
        """Is communicator closed"""
        return self._close_init.locked()

    def _recive_response(self) -> None:
        """Recive one response from communication"""
        while self._requests:
            try:
                response = self._comm.get().data
            except Exception as exc:
                warnings.warn(repr(exc), RuntimeWarning)
                continue

            match response:
                case proto.OperationResponse():
                    pass

                case _:
                    raise RuntimeError(f"Unknown response {response}")

            self._responses[response.id] = response.data

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
                    if isinstance(response, proto.RemoteException):
                        request._callback = Request._callback  # Remove callback
                    request._state = RequestState.RES
                    future = self._resolve_queue.submit(request._callback, response)
                    future.add_done_callback(request._resolve)

                case _:
                    raise RuntimeError(f"Invalid request state {request._state}")

    def submit(self, op: proto.OperationRequest, callback: abc.Callable = Request._callback) -> Request:
        """
        Schedule a new operation

        Operation can be a user defined instance, allowing for custom operations.
        """

        # Create appropriate request according to participation
        op = copy.replace(op, **{
            "group": op.group,
            "ctx": op.ctx if self.rank == op.group.root else None,
            "data": op.data if self.rank in op.group.src else None
        })

        # Setup request
        request = Request()
        request._callback = callback
        asynctools.future_set_running(request.future)

        # Check if operation valid
        if self.rank not in op.group:
            asynctools.future_set_exception(request.future, RuntimeError("Tried to schedule operation without participating"))
            return request

        # Schedule operation
        if self.rank in op.group.dst:
            self._requests[op.id] = request

        future = self._comm.put(op)

        if self.rank in op.group.dst:
            future.add_done_callback(lambda future: future.exception() and request._resolve(future))
            self._recv_queue.submit(self._recive_response).add_done_callback(asynctools.future_warn_exception)
        else:
            future.add_done_callback(request._resolve)

        return request

    @functools.cached_property
    def _comm(self) -> nq.Communicator:
        """Communication connection"""
        # NOTE: Lazily initialized, prevent module imports execution
        with self._comm_lock:
            if "_comm" in self.__dict__:
                comm = self.__dict__["_comm"]
            else:
                comm = self.__dict__["_comm"] = self._new_comm()
        return comm

    def _new_comm(self) -> nq.Communicator:
        """Create a new communication and inizialize it"""

        # If requested, start a local server
        if rc.init:
            if self.rank == 0:
                from pympi.server import background_server
                self._server = background_server()

            # Allow some time for server startup
            from time import sleep
            sleep(rc.wait)

        state = proto.RankInit(rank=self.rank)
        try:
            comm = nq.new(protocol=rc.proto, purpose=nq.Purpose.CLIENT, options=self._comm_options)
            comm.put(state)
            while True:
                response = comm.get().data

                match response:
                    case proto.StateResponse():
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

    def _close_comm(self, comm: nq.Communicator) -> None:
        """Fianlize a communication object"""
        state = proto.RankFinalize()
        try:
            comm.put(state)
            while True:
                response = comm.get().data

                match response:
                    case proto.StateResponse():
                        pass
                    case _:
                        warnings.warn(f"Unknown response {response}", RuntimeWarning)
                        continue  # response lost

                if response.size == 0:
                    break
        finally:
            comm.close()

        # If requested, stop a local server
        if rc.init:
            if self.rank == 0:
                self._server.result()

            # Allow some time for server shutdown
            from time import sleep
            sleep(rc.wait)

    def _close(self) -> None:
        """Communicator finalizer"""
        # Close comunicator if initialized
        with self._comm_lock:
            if "_comm" in self.__dict__:
                self.barrier()

            self._recv_queue.shutdown()
            self._resolve_queue.shutdown()

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
        return rc.size

    @property
    def rank(self) -> proto.Rank:
        """Communication identifier"""
        return rc.rank

    def Get_rank(self) -> int:
        """Return the rank of this process in a communicator."""
        return self.rank

    def Get_size(self) -> int:
        """Return the number of processes in a communicator."""
        return self.size

    # Barrier synchronization
    def ibarrier(self) -> Request[None]:
        """Nonblocking Barrier synchronization."""
        return self.iallreduce(None, proto.ReduceOperation.LAND)

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
    def ibcast[T](self, obj: T, root: proto.Rank = 0) -> Request[T]:
        """Broadcast."""
        ctx = proto.BroadcastContext()
        group = ctx.group(root=root, size=self.size)
        op = proto.OperationRequest(group=group, ctx=ctx, data=obj)
        return self.submit(op=op)

    def bcast[T](self, obj: T, root: proto.Rank = 0) -> T:
        """Broadcast."""
        return self.ibcast(obj=obj, root=root).wait()

    def Ibcast[T: abc.Buffer](self, buf: T, root: proto.Rank = 0) -> Request[None]:
        """Nonblocking Gather to All."""
        ctx = proto.BroadcastContext()
        group = ctx.group(root=root, size=self.size)

        def callback(result: T) -> None:
            with byteview(result) as src, byteview(buf) as dst:
                dst[:] = src

        op = proto.OperationRequest(group=group, ctx=ctx, data=buf)
        return self.submit(op=op, callback=callback)

    def Bcast[T: abc.Buffer](self, buf: T, root: proto.Rank = 0) -> None:
        """Gather to All."""
        return self.Ibcast(buf=buf, root=root).wait()

    # Gather to All
    def iallgather[T](self, sendobj: T) -> Request[list[T]]:
        """Nonblocking Gather to All."""
        ctx = proto.AllGatherContext()
        group = ctx.group(size=self.size)
        op = proto.OperationRequest(group=group, ctx=ctx, data=sendobj)
        return self.submit(op=op)

    def allgather[T](self, sendobj: T) -> list[T]:
        """Gather to All."""
        return self.iallgather(sendobj=sendobj).wait()

    # Reduce to All
    def iallreduce[T](self, sendobj: T, op: proto.ReduceOperation = proto.ReduceOperation.SUM) -> Request[T]:
        """Reduce to All."""
        ctx = proto.AllReduceContext(op=op)
        group = ctx.group(size=self.size)
        operation = proto.OperationRequest(group=group, ctx=ctx, data=sendobj)
        return self.submit(op=operation)

    def allreduce[T](self, sendobj: T, op: proto.ReduceOperation = proto.ReduceOperation.SUM) -> T:
        """Reduce to All."""
        return self.iallreduce(sendobj=sendobj, op=op).wait()

    def Iallreduce[T: abc.Buffer](self, sendbuf: T | typing.Literal[InPlace.IN_PLACE], recvbuf: T, op: proto.ReduceOperation = proto.ReduceOperation.SUM) -> Request[None]:
        """Nonblocking Reduce to All."""
        if sendbuf is InPlace.IN_PLACE:
            sendbuf = recvbuf

        def callback(result: T) -> None:
            with byteview(result) as src, byteview(recvbuf) as dst:
                dst[:] = src

        ctx = proto.AllReduceContext(op=op)
        group = ctx.group(size=self.size)
        operation = proto.OperationRequest(group=group, ctx=ctx, data=sendbuf)
        return self.submit(op=operation, callback=callback)

    def Allreduce[T: abc.Buffer](self, sendbuf: T | typing.Literal[InPlace.IN_PLACE], recvbuf: T, op: proto.ReduceOperation = proto.ReduceOperation.SUM) -> None:
        """Reduce to All."""
        self.Iallreduce(sendbuf=sendbuf, recvbuf=recvbuf, op=op).wait()

    # Reduce to All (with steps)
    def _phased_allreduce[T](self, sendobj: T, op: proto.ReduceOperation = proto.ReduceOperation.SUM) -> T:
        """Reduce to All (with steps)."""
        ctx = proto.AllPhasedReduceContext(op=op)

        for phase in itertools.count():
            group = ctx.group(rank=self.rank, size=self.size, phase=phase)
            operation = proto.OperationRequest(group=group, ctx=ctx, data=sendobj)
            sendobj = self.submit(op=operation).wait()
            if len(group.dst) == self.size:
                break

        return sendobj

    # Scatter
    def iscatter[T](self, sendobj: abc.Sequence[T], root: proto.Rank = 0) -> Request[T]:
        """Nonblocking Scatter."""
        ctx = proto.ScatterContext()
        group = ctx.group(root=root, size=self.size)
        op = proto.OperationRequest(group=group, ctx=ctx, data=sendobj)
        return self.submit(op=op)

    def scatter[T](self, sendobj: abc.Sequence[T], root: proto.Rank = 0) -> T:
        """Scatter."""
        return self.iscatter(sendobj=sendobj, root=root).wait()

    # All to All Scatter/Gather
    def ialltoall[T](self, sendobj: abc.Sequence[T]) -> Request[list[T]]:
        """All to All Scatter/Gather."""
        ctx = proto.AllToAllContext()
        group = ctx.group(size=self.size)
        op = proto.OperationRequest(group=group, ctx=ctx, data=sendobj)
        return self.submit(op=op)

    def alltoall[T](self, sendobj: abc.Sequence[T]) -> list[T]:
        """All to All Scatter/Gather."""
        return self.ialltoall(sendobj=sendobj).wait()

    # Gather to All
    def igather[T](self, sendobj: T, root: proto.Rank = 0) -> Request[list[T]]:
        """Nonblocking Gather."""
        ctx = proto.GatherContext()
        group = ctx.group(root=root, size=self.size)
        op = proto.OperationRequest(group=group, ctx=ctx, data=sendobj)
        return self.submit(op=op)

    def gather[T](self, sendobj: T, root: proto.Rank = 0) -> list[T]:
        """Gather."""
        return self.igather(sendobj=sendobj, root=root).wait()

    # Reduce
    def ireduce[T](self, sendobj: T, op: proto.ReduceOperation = proto.ReduceOperation.SUM, root: proto.Rank = 0) -> Request[T]:
        """Reduce to All."""
        ctx = proto.ReduceContext(op=op)
        group = ctx.group(root=root, size=self.size)
        operation = proto.OperationRequest(group=group, ctx=ctx, data=sendobj)
        return self.submit(op=operation)

    def reduce[T](self, sendobj: T, op: proto.ReduceOperation = proto.ReduceOperation.SUM, root: proto.Rank = 0) -> T:
        """Reduce to All."""
        return self.ireduce(sendobj=sendobj, op=op, root=root).wait()

    def Ireduce[T: abc.Buffer](self, sendbuf: T | typing.Literal[InPlace.IN_PLACE], recvbuf: T, op: proto.ReduceOperation = proto.ReduceOperation.SUM, root: proto.Rank = 0) -> Request[None]:
        """Nonblocking Reduce to All."""
        if sendbuf is InPlace.IN_PLACE:
            sendbuf = recvbuf

        def callback(result: T) -> None:
            with byteview(result) as src, byteview(recvbuf) as dst:
                dst[:] = src

        ctx = proto.ReduceContext(op=op)
        group = ctx.group(root=root, size=self.size)
        operation = proto.OperationRequest(group=group, ctx=ctx, data=sendbuf)
        return self.submit(op=operation, callback=callback)

    def Reduce[T: abc.Buffer](self, sendbuf: T | typing.Literal[InPlace.IN_PLACE], recvbuf: T, op: proto.ReduceOperation = proto.ReduceOperation.SUM, root: proto.Rank = 0) -> None:
        """Reduce to All."""
        self.Ireduce(sendbuf=sendbuf, recvbuf=recvbuf, op=op, root=root).wait()

    # Send
    def isend[T](self, obj: T, dest: proto.Rank, tag: proto.Tag = 0) -> Request[None]:
        """Nonblocking Send in standard mode."""
        if dest == self.rank:
            raise ValueError("Self send not supported")

        if tag:
            raise ValueError("Tags are not supported")

        ctx = proto.SendRecvContext(tag=tag)
        group = ctx.group(src=self.rank, dst=dest)
        op = proto.OperationRequest(group=group, ctx=ctx, data=obj)
        return self.submit(op=op)

    def send[T](self, obj: T, dest: proto.Rank, tag: proto.Tag = 0) -> None:
        """Send in standard mode."""
        return self.isend(obj=obj, dest=dest, tag=tag).wait()

    def Isend[T: abc.Buffer](self, buf: T, dest: proto.Rank, tag: proto.Tag = 0) -> Request[None]:
        """Nonblocking Send in standard mode."""
        return self.isend(obj=buf, dest=dest, tag=tag)

    def Send[T: abc.Buffer](self, buf: T, dest: proto.Rank, tag: proto.Tag = 0) -> None:
        """Send in standard mode."""
        return self.Isend(buf=buf, dest=dest, tag=tag).wait()

    # Receive
    def irecv[T: abc.Buffer](self, buf: T | None = None, source: proto.Rank = ANY_SOURCE, tag: proto.Tag = ANY_TAG, status=None) -> Request[T]:
        """Nonblocking Receive."""
        if source == self.rank:
            raise ValueError("Self source not supported")

        if source == ANY_SOURCE:
            raise ValueError("Any source is not supported")

        if tag:
            raise ValueError("Tags are not supported")

        if status:
            raise ValueError("Status are not supported")

        def callback(result: T):
            if buf is None:
                return result
            with byteview(result) as src, byteview(buf) as dst:
                dst[:] = src

        ctx = proto.SendRecvContext(tag=tag)
        group = ctx.group(src=source, dst=self.rank)
        op = proto.OperationRequest(group=group, ctx=ctx)
        return self.submit(op=op, callback=callback)

    def recv[T: abc.Buffer](self, buf: T | None = None, source: proto.Rank = ANY_SOURCE, tag: proto.Tag = ANY_TAG, status=None) -> T:
        """Nonblocking Receive."""
        return self.irecv(buf=buf, source=source, tag=tag, status=status).wait()

    # Send and Recevie
    def isendrecv[S, R: abc.Buffer](self, sendobj: S, dest: proto.Rank, sendtag: proto.Rank = 0, recvbuf: R | None = None, source: proto.Rank = ANY_SOURCE, recvtag: proto.Tag = ANY_TAG, status=None) -> Request[R]:
        """Nonblocking Send and Receive."""
        send = self.isend(obj=sendobj, dest=dest, tag=sendtag)
        recv = self.irecv(buf=recvbuf, source=source, tag=recvtag, status=status)

        def callback(future: Future[None]) -> None:
            if send.future.exception():
                request._resolve(send.future)  # type: ignore
            request._resolve(recv.future)

        request = Request[R]()
        future = asynctools.merge_futures([send.future, recv.future])
        future.add_done_callback(callback)
        return request

    def sendrecv[S, R: abc.Buffer](self, sendobj: S, dest: proto.Rank, sendtag: proto.Rank = 0, recvbuf: R | None = None, source: proto.Rank = ANY_SOURCE, recvtag: proto.Tag = ANY_TAG, status=None) -> R:
        """Nonblocking Send and Receive."""
        return self.isendrecv(sendobj=sendobj, dest=dest, sendtag=sendtag, recvbuf=recvbuf, source=source, recvtag=recvtag, status=status).wait()

    # Send and Recevie
    def Isendrecv[S: abc.Buffer, R: abc.Buffer](self, sendbuf: S, dest: proto.Rank, sendtag: proto.Rank = 0, recvbuf: R | None = None, source: proto.Rank = ANY_SOURCE, recvtag: proto.Tag = ANY_TAG, status=None) -> Request[R]:
        """Nonblocking Send and Receive."""
        send = self.Isend(buf=sendbuf, dest=dest, tag=sendtag)
        recv = self.irecv(buf=recvbuf, source=source, tag=recvtag, status=status)

        def callback(future: Future[None]) -> None:
            if send.future.exception():
                request._resolve(send.future)  # type: ignore
            request._resolve(recv.future)

        request = Request[R]()
        future = asynctools.merge_futures([send.future, recv.future])
        future.add_done_callback(callback)
        return request

    def Sendrecv[S: abc.Buffer, R: abc.Buffer](self, sendbuf: S, dest: proto.Rank, sendtag: proto.Rank = 0, recvbuf: R | None = None, source: proto.Rank = ANY_SOURCE, recvtag: proto.Tag = ANY_TAG, status=None) -> R:
        """Nonblocking Send and Receive."""
        return self.Isendrecv(sendbuf=sendbuf, dest=dest, sendtag=sendtag, recvbuf=recvbuf, source=source, recvtag=recvtag, status=status).wait()


# Exports
IN_PLACE = InPlace.IN_PLACE

MAX = proto.ReduceOperation.MAX
MIN = proto.ReduceOperation.MIN
SUM = proto.ReduceOperation.SUM
PROD = proto.ReduceOperation.PROD
LAND = proto.ReduceOperation.LAND
BAND = proto.ReduceOperation.BAND
LOR = proto.ReduceOperation.LOR
BOR = proto.ReduceOperation.BOR
LXOR = proto.ReduceOperation.LXOR
BXOR = proto.ReduceOperation.BXOR
MINLOC = proto.ReduceOperation.MINLOC
MAXLOC = proto.ReduceOperation.MAXLOC

COMM_WORLD = Comm(rc.comm)

# Best effort finalizer
try:
    threading._register_atexit(Finalize)  # type: ignore (private implementation dependant)
except AttributeError:
    atexit.register(Finalize)
