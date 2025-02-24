"""Message Passing Interface (server)"""

# NOTE: Module considerations
#
# Communications are lazily initialized to prevent module imports execution

# TODO: Revise sending lock usage, could only lock per commuincaiton group

import uuid
import functools
import threading
import dataclasses
from collections import deque
from dataclasses import dataclass
from argparse import ArgumentParser, Namespace
from concurrent.futures import ThreadPoolExecutor

from bidict import bidict

from pydtnn import comms
from pydtnn.libs.mpi import comm as mpi_comm


__all__ = (
    "Server",
    "start_local_server"
)


# Argument pasrser
arg_parser = ArgumentParser(
    prog="mpi_server",
    description="MPI server"
)
arg_parser.add_argument("-np", dest="size", type=int, default=4)
arg_parser.add_argument("--oneshot", action="store_true")


@dataclass(slots=True)
class Operation[T: mpi_comm.OperationRequest]:
    """MPI Operation"""
    comm: mpi_comm.CommmunicationGroup
    requests: dict[mpi_comm.Rank, T] = dataclasses.field(default_factory=dict)
    responses: list | None = None

    def put(self, rank: mpi_comm.Rank, request: T) -> bool:
        return request is self.requests.setdefault(rank, request)

    def full(self) -> bool:
        return len(self.requests) == len(self.comm.src)


class Server:
    """MPI server"""

    def __init__(self, thread_pool: ThreadPoolExecutor) -> None:
        """Server initialization"""
        super().__init__()

        # State
        self._shutdown = False
        self._pool = thread_pool
        self._comm_lock = threading.Lock()
        self._response_count = threading.Semaphore(value=0)

        self._state_lock = threading.Lock()
        self._state = dict[mpi_comm.CommmunicationGroup, deque[Operation]]()

        self._peers_lock = threading.Lock()
        self._peers = bidict[mpi_comm.Rank, uuid.UUID]()

    @property
    def _size(self):
        """Get the approximate number of clients"""
        return len(self._peers)

    def _submit(self, fn, /, *args, **kwargs):
        """Process in the pool with exception handeling"""
        future = self._pool.submit(fn, *args, **kwargs)
        future.add_done_callback(lambda future: future.result())
        return future

    @functools.cached_property
    def _comm(self) -> comms.Communication:
        """Communication connection"""
        # NOTE: Lazily initialized, prevent module imports execution
        with self._comm_lock:
            if comm := self.__dict__.get("_comm"):
                pass
            else:
                addr = mpi_comm.get_addr()
                port = mpi_comm.get_port()
                comm = comms.Server(addr=addr, port=port)
                self._comm = comm
        return comm

    def __enter__(self):
        """Context manager start"""
        return self

    def __exit__(self, cls, exc, tb):
        """Context manager exit"""
        self.shutdown()

    def __del__(self) -> None:
        """Best effort finalizer"""
        try:
            self.shutdown()
        except:  # noqa: E722
            pass

    def serve_forever(self) -> None:
        """Handle until shutdown"""
        while self._shutdown:
            self.serve_util_finalize()

    def serve_util_finalize(self) -> None:
        """Handle until clients call finalize"""
        self._submit(self._handle_operations_responses)
        self._handle_requests()

    def _handle_requests(self) -> None:
        """Handle requests forever"""

        while not self._shutdown:
            message = self._comm.get()
            request = message.obj

            # Handle request
            match request:
                case mpi_comm.StateRequest():
                    self._handle_state_request(message)
                case mpi_comm.OperationRequest():
                    self._handle_operation_request(message)
                case _:
                    raise RuntimeError(f"Unknown request type {request}")

            # Finish if idle
            if self._size == 0:
                break

    def _handle_state_request(self, message: comms.Message[mpi_comm.StateRequest]) -> None:
        """Handle an state request"""
        request = message.obj

        match request:
            case mpi_comm.InitRequest():
                self._handle_init(message)  # type: ignore (not inferred by typecheker)
            case mpi_comm.FinalizeRequest():
                self._handle_finalize(message)  # type: ignore (not inferred by typecheker)
            case _:
                raise RuntimeError(f"Unknown state type {request}")

    def _handle_init(self, message: comms.Message[mpi_comm.InitRequest]) -> None:
        """Initialize."""
        # Request context
        peer = message.peer
        request = message.obj
        rank = request.rank
        size = request.size

        # Thread-safe client setup
        with self._peers_lock:
            self._peers[rank] = peer

        # Syncronize clients when all ready
        if self._size == size:
            self._comm.put(None)

    def _handle_finalize(self, message: comms.Message[mpi_comm.FinalizeRequest]) -> None:
        """Terminate."""
        # Request context
        peer = message.peer
        rank = self._peers.inverse[peer]

        # Thread-safe client taredown
        with self._peers_lock:
            del self._peers[rank]

        # Syncronize clients when all ready
        if self._size == 0:
            self._comm.put(None)

    def _handle_operation_request(self, message: comms.Message[mpi_comm.OperationRequest]) -> None:
        """Handle an operation request"""
        # Operation context
        peer = message.peer
        request = message.obj
        rank = self._peers.inverse[peer]

        # Queue operation
        with self._state_lock:
            queue = self._state.setdefault(request.comm, deque())

            for operation in queue:
                if operation.put(rank, request):
                    break
            else:
                operation = Operation(comm=request.comm)
                put = operation.put(rank, request)
                assert put, "Failed to put on a empty operation"
                queue.append(operation)

        # Start operation
        if operation.full():
            self._submit(self._handle_operation, operation)

    def _handle_operations_responses(self) -> None:
        """Handle responses forever"""
        while not self._shutdown:
            # Wait for any response
            self._response_count.acquire()
            self._response_count.release()

            # Send and consume
            for operation in self._pop_operations_responded():
                self._response_count.acquire()
                self._send_operation(operation)

            # Finish if idle
            if self._size == 0:
                break

    def _pop_operations_responded(self) -> list[Operation]:
        """Get and remove responded operations from state"""
        operations = list[Operation]()

        with self._state_lock:
            for group, queue in list(self._state.items()):
                for operation in list(queue):
                    if operation.responses is not None:
                        queue.remove(operation)
                        operations.append(operation)
                    else:
                        break

                if len(queue) == 0:
                    del self._state[group]

        return operations

    def _send_operation(self, operation: Operation) -> None:
        """Send a operation response to the clients"""
        assert operation.responses is not None, f"Sending in progress operation {operation}"

        for obj in operation.responses:
            response = mpi_comm.OperationResponse(dst=operation.comm.dst, obj=obj)
            self._comm.put(response, *(
                self._peers[rank]
                for rank in operation.comm.dst
            ))

    def _handle_operation(self, operation: Operation[mpi_comm.OperationRequest]) -> None:
        """Dispatch operation to relevant handler"""
        context = next(iter(operation.requests.values()))

        match context:
            case mpi_comm.BroadcastRequest():
                handler = self._handle_broadcast
            case mpi_comm.AllGatherRequest():
                handler = self._handle_allgather
            case mpi_comm.AllReduceRequest():
                handler = self._handle_allreduce
            case mpi_comm.AllPhasedReduceRequest():
                handler = self._handle_allphasedreduce
            case _:
                raise RuntimeError(f"Unknown operation type {context}")

        handler(operation)  # type: ignore (not inferred by typecheker)
        self._response_count.release()

    def _handle_broadcast(self, operation: Operation[mpi_comm.BroadcastRequest]) -> None:
        """Broadcast."""
        response = operation.requests[0].obj
        operation.responses = [response]

    def _handle_allgather(self, operation: Operation[mpi_comm.AllGatherRequest]) -> None:
        """Gather to All."""
        rank_requests = sorted(
            operation.requests.items(),
            key=lambda rank_request: rank_request[0]
        )

        operation.responses = [
            request.obj
            for _, request in rank_requests
        ]

    def _handle_allreduce(self, operation: Operation[mpi_comm.AllReduceRequest]) -> None:
        """Reduce to All."""
        response = sum(msg.obj for msg in operation.requests.values())
        operation.responses = [response]

    def _handle_allphasedreduce(self, operation: Operation[mpi_comm.AllReduceRequest]) -> None:
        """Reduce to All (with steps)."""
        response = sum(msg.obj for msg in operation.requests.values())
        operation.responses = [response]

    def shutdown(self) -> None:
        """Close the server"""
        if comm := self.__dict__.pop("_comm", None):
            comm.close()

        if self._shutdown:
            return
        self._shutdown = True

        self._response_count.release()


def start_local_server() -> None:
    """Start a local background server"""
    from threading import Thread
    pool = ThreadPoolExecutor(max_workers=mpi_comm.get_size())
    server = Server(pool)

    # Ensure connection is setup
    server._comm

    # Serve and finalize handler
    def serve_oneshot():
        server.serve_util_finalize()
        server.shutdown()
    Thread(target=serve_oneshot).start()


def main(config: Namespace) -> None:
    """Application entrypoint"""
    with ThreadPoolExecutor(max_workers=config.size) as pool:
        with Server(pool) as server:
            if config.oneshot:
                server.serve_util_finalize()
            else:
                server.serve_forever()


if __name__ == "__main__":
    main(arg_parser.parse_args())
