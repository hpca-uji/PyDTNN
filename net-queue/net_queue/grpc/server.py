"""gRPC server"""

import grpc
import traceback
from collections import abc

from net_queue import server
from net_queue.grpc import Protocol
from net_queue import CommunicatorOptions


__all__ = (
    "Communicator",
)


# Sentinel objects
END_COMM = None


class Communicator(Protocol[str], server.Server[str]):
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
        self._server.add_registered_method_handlers(service_name="grpc", method_handlers={"comm": handler})  # type: ignore

        config: abc.MutableMapping = {
            "address": str(self.options.netloc)
        }

        if self.options.security:
            if self.options.security.certificate is None or self.options.security.key is None:
                raise RuntimeError("SSL certificate or key not provided")
            config["server_credentials"] = grpc.ssl_server_credentials([
                (self.options.security.key.read_bytes(), self.options.security.certificate.read_bytes()),  # type: ignore
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
        comm = context.peer()
        peer = self._set_default_peer(comm)
        state = self._states[peer]

        # Message streaming
        yield from self._put_flush(peer)
        for data in messages:
            state.get_write(data)
            self._process_gets(peer)
            peer = state.peer

        if not state.state and state.put_empty():
            self._connection_fin(comm)

    def _close(self) -> None:
        """Close the server"""

        # Wait peers to drain
        with self._lock:
            while self._comms:
                self._lock.wait()

        # Close resources
        # Allow some time for RPC taredown
        self._server.stop(grace=0.5)
        self._pool.shutdown()

        super()._close()
