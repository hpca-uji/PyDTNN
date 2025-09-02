"""gRPC server"""

import grpc
import threading
import traceback
from collections import abc

from pydtnn import comms
from pydtnn.comms import server
from pydtnn.utils import UUID_MAX
from pydtnn.comms.grpc import Protocol
from pydtnn.comms import CommunicatorOptions


__all__ = (
    "Server",
)


# Sentinel objects
END_COMM = None


class Server(Protocol, server.Server[str]):
    """gRPC server"""

    def __init__(self, options: CommunicatorOptions = CommunicatorOptions()) -> None:
        """Server initialization"""
        super().__init__(options)

        # gRPC
        self._server = grpc.server(
            thread_pool=self._pool,
            options=list(self._grpc_options.items()),
            compression=self._compression
        )
        handler = grpc.stream_stream_rpc_method_handler(behavior=self._com, request_deserializer=lambda x: x, response_serializer=bytes)
        self._server.add_registered_method_handlers(service_name="grpc", method_handlers={"com": handler})  # type: ignore

        config: abc.MutableMapping = {
            "address": str(self._options.netloc)
        }

        if comms.SSL:
            config["server_credentials"] = grpc.ssl_server_credentials([
                (comms.SSL_KEY.read_bytes(), comms.SSL_CERT.read_bytes()),  # type: ignore
            ])
            self._server.add_secure_port(**config)
        else:
            self._server.add_insecure_port(**config)

        self._server.start()

    def _com(self, messages: abc.Iterable[abc.Buffer], context: grpc.ServicerContext) -> abc.Iterable[abc.Buffer]:
        try:
            yield from self._handle_connection(messages, context)
        except Exception as exc:
            traceback.print_exception(exc)
            context.set_code(grpc.StatusCode.INTERNAL)

    def _handle_connection(self, messages: abc.Iterable[abc.Buffer], context: grpc.ServicerContext) -> abc.Iterable[abc.Buffer]:
        """Client to server communication"""
        # NOTE: communication thread
        sock = context.peer()
        peer = self._get_peer(sock)
        state = self._get_state(peer)

        # Message streaming
        yield from self._s2m(state)
        for data in self._m2d(messages):
            state.get_write(data)
            peer = self._get_flush(peer)

        if not state.state and state.put_empty():
            self._fin(peer)

    def _close(self) -> None:
        """Close the server"""

        # Wait peers to drain
        with self._lock:
            while self._peers:
                self._lock.wait()

        # Close resources
        # Allow some time for RPC taredown
        self._server.stop(grace=0.5)
        self._pool.shutdown()

        # Unlock inflight external API
        for _ in range(threading.active_count()):
            self._get_event.put(UUID_MAX)
        super()._close()
