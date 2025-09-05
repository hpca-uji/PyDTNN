"""MQTT client"""

import uuid
from concurrent.futures import Future

import paho.mqtt.client as mqtt_client

from pydtnn.comms import client
from pydtnn.comms.mqtt import Protocol
from pydtnn.utils.io_stream import Stream
from pydtnn.comms import CommunicatorOptions


__all__ = (
    "Client",
)


# Sentinel objects
END_COMM = b""


class Client(Protocol[str], client.Client[str]):
    """MQTT client"""

    def __init__(self, options: CommunicatorOptions = CommunicatorOptions()) -> None:
        """Client initialization"""
        super().__init__(options)

        # MQTT
        self._sock = self._id.hex
        self._register_handler(topic=f"s2c/{self._sock}", handler=self._handle_message)

        self._ini(self._sock)

    def _handle_message(self, client: mqtt_client.Client, userdata, message: mqtt_client.MQTTMessage) -> None:
        """Broker message handler"""
        sock = self._peer(message)
        peer = self._get_peer(sock)
        state = self._state[peer]
        state.get_buffer.write(message.payload)
        self._process_session(peer)

        if not state.state and state.put_empty():
            self._fin(sock)

    def _put(self, stream: Stream, peer: uuid.UUID) -> Future[None]:
        """Put stream into queue and notify"""
        future = super()._put(stream, peer)
        self._pool.submit(self._c2s, self._sock).add_done_callback(lambda future: future.result())
        return future

    def _c2s(self, sock: str):
        peer = self._get_peer(sock)
        state = self._state[peer]

        state.put_flush()
        while not state.put_buffer.empty():
            with state.put_read(self._options.connection.max_size) as view:
                self._publish(f"c2s/{self._id.hex}", bytes(view))

    def _close(self) -> None:
        """Close the client"""
        sock = self._sock
        peer = self._get_peer(sock)
        state = self._state[peer]
        self._put(self._session_fin(state), peer)

        # Request loop thread to stop
        self._pool.shutdown()
        self._stop_loop()

        super()._close()
