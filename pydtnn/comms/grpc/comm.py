"""gRPC communication"""

import grpc
import threading
from queue import Empty, SimpleQueue
from concurrent.futures import ThreadPoolExecutor
from pydtnn.comms.grpc import Protocol, grpc_pb2, grpc_pb2_grpc


__all__ = (
    "Server",
    "Client"
)


class Server(Protocol):
    """gRPC server"""

    def __init__(self) -> None:
        """Server initialization"""
        super().__init__()

        # State
        self._lock = threading.Lock()
        self._requests = SimpleQueue[grpc_pb2.Message]()
        self._responses = dict[str, SimpleQueue[grpc_pb2.Message]]()

        # gRPC
        thread_pool = ThreadPoolExecutor(max_workers=1)
        self._server = grpc.server(
            thread_pool=thread_pool,
            compression=self._compression
        )
        grpc_pb2_grpc.add_gRPCServicer_to_server(servicer=self, server=self._server)
        self._server.add_insecure_port(address=self._netloc)
        self._server.start()

    def _syc(self, request: grpc_pb2.Message, context: grpc.ServicerContext) -> grpc_pb2.Message:
        """Client connection startup"""
        with self._lock:
            self._responses[context.peer()] = SimpleQueue()
        return grpc_pb2.Message()

    def _c2s(self, request: grpc_pb2.Message, context: grpc.ServicerContext) -> grpc_pb2.Message:
        """Client to server communication"""
        self._requests.put(request)
        return grpc_pb2.Message()

    def _s2c(self, request: grpc_pb2.Message, context: grpc.ServicerContext) -> grpc_pb2.Message:
        """Server to client communication"""
        try:
            return self._responses[context.peer()].get_nowait()
        except Empty:
            context.set_code(grpc.StatusCode.UNAVAILABLE)
            return grpc_pb2.Message()

    def _fin(self, request: grpc_pb2.Message, context: grpc.ServicerContext) -> grpc_pb2.Message:
        """Client connection finalizer"""
        with self._lock:
            del self._responses[context.peer()]
        return grpc_pb2.Message()

    def get(self):
        """Get data from a client"""
        msg = self._requests.get()
        obj = self._deserialize(msg.data)
        return obj

    def put(self, obj) -> None:
        """Publish data to clients"""
        data = self._serialize(obj)
        msg = grpc_pb2.Message(data=data)
        with self._lock:
            for queue in self._responses.values():
                queue.put(msg)

    def close(self) -> None:
        """Close the server"""
        self._server.stop(grace=None)
        super().close()


class Client(Protocol):
    """gRPC client"""

    def __init__(self) -> None:
        """Client initialization"""
        self._channel = grpc.insecure_channel(
            target=self._netloc,
            compression=self._compression
        )
        self._client = grpc_pb2_grpc.gRPCStub(self._channel)
        self._client._syc(grpc_pb2.Message())

    def put(self, obj) -> None:
        """Publish data to server"""
        data = self._serialize(obj)
        msg = grpc_pb2.Message(data=data)
        self._client._c2s(msg)

    def get(self):
        """Get server data"""
        req = grpc_pb2.Message()
        while True:
            try:
                res: grpc_pb2.Message = self._client._s2c(req)
            except grpc.RpcError as exc:
                if exc.code() is grpc.StatusCode.UNAVAILABLE:  # type: ignore
                    pass
            else:
                obj = self._deserialize(res.data)
                return obj

    def close(self) -> None:
        """Close the client"""
        self._client._fin(grpc_pb2.Message())
        self._channel.close()
        super().close()
