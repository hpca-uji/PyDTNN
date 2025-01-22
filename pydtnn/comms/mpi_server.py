"""Message Passing Interface (server)"""

import typing
import threading
import functools
from pydtnn import comms
from queue import SimpleQueue
from argparse import ArgumentParser, Namespace
from concurrent.futures import ThreadPoolExecutor


__all__ = (
    "Server",
)


# Argument pasrser
arg_parser = ArgumentParser(
    prog="mpi_server",
    description="MPI server"
)
arg_parser.add_argument("-np", dest="size", type=int, default=4)
arg_parser.add_argument("-a", dest="addr", type=str, default="localhost")
arg_parser.add_argument("-p", dest="port", type=str, default=None)


class Server:
    """MPI server"""

    def __init__(self, thead_pool: ThreadPoolExecutor) -> None:
        """Server initialization"""
        super().__init__()

        # State
        self._pool = thead_pool
        self._lock = threading.Lock()
        self._queue = SimpleQueue[comms.MPI_Request]()

    @functools.cached_property
    def _comm(self) -> comms.Communication:
        """Communication connection"""
        # Lazily initialized to prevent module imports execution
        return comms.Server()

    def _get(self) -> comms.MPI_Request:
        """Get request from clients"""
        return self._queue.get()

    def _put(self, obj) -> None:
        """Publish response to clients"""
        response = comms.MPI_Response(obj)
        self._comm.put(response)

    def _handle(self, request: comms.MPI_Request) -> None:
        """Client request handler"""
        # Store data
        self._queue.put(request)

        # Compute (if nobody else)
        if not self._lock.acquire(blocking=False):
            return
        try:
            name = request.op.name.lower()
            handler = getattr(self, name)
            handler(request)
        finally:
            self._lock.release()

    def serve_forever(self) -> None:
        """Serve requests forever using worker pool"""
        try:
            while True:
                request = self._comm.get()
                self._pool.submit(self._handle, request)
        finally:
            self.close()

    def close(self) -> None:
        """Close the server"""
        if "_comm" in self.__dict__:
            self._comm.close()

    def __del__(self) -> None:
        """Best effort finalizer"""
        try:
            self.close()
        except:  # noqa: E722
            pass

    def broadcast(self, context: comms.MPI_Request) -> None:
        """Broadcast."""
        self._put(self._get().obj)

    def gather(self, context: comms.MPI_Request) -> None:
        """Gather to All."""
        reqs = sorted(
            (self._get() for _ in range(context.size)),
            key=lambda req: req.rank
        )

        for req in reqs:
            self._put(req.obj)

    def reduce(self, context: comms.MPI_Request) -> None:
        """Reduce to All."""
        self._put(sum(self._get().obj for _ in range(context.size)))


def main(*args: str) -> None:
    """Application entrypoint"""
    config = arg_parser.parse_args(args)
    config = typing.cast(Namespace, config)
    thread_pool = ThreadPoolExecutor(max_workers=config.size)
    comms.Server._addr = config.addr  # type: ignore
    if config.port:
        comms.Server._port = config.port
    Server(thread_pool).serve_forever()


if __name__ == "__main__":
    import sys
    main(*sys.argv[1:])
