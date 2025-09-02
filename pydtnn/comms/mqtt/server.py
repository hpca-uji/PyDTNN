"""MQTT server"""

import uuid
import threading
from concurrent.futures import Future

import paho.mqtt.client as mqtt_client

from pydtnn.comms import server
from pydtnn.utils import UUID_MAX
from pydtnn.comms.mqtt import Protocol
from pydtnn.utils.io_stream import Stream
from pydtnn.comms import CommunicatorOptions


__all__ = (
    "Server",
)


# Sentinel objects
END_COMM = b""


class Server(Protocol, server.Server[str]):
    """MQTT server"""

    def __init__(self, options: CommunicatorOptions = CommunicatorOptions()) -> None:
        """Server initialization"""
        super().__init__(options)

        # MQTT
        self._register_handler(topic="c2s/+", handler=self._c2s)

    def _extra_ini(self, peer: uuid.UUID) -> None:
        sock = self._peers.pop(peer)
        state = self._state.pop(peer)
        peer = uuid.UUID(hex=sock)
        self._peers[peer] = sock
        self._state[peer] = state

    def _c2s(self, client: mqtt_client.Client, userdata, mqtt_message: mqtt_client.MQTTMessage) -> None:
        """Client message handler"""
        # NOTE: communication thead
        sock = self._peer(mqtt_message)
        peer = self._get_peer(sock)
        state = self._get_state(peer)

        data = mqtt_message.payload
        state.get_write(data)
        peer = self._get_flush(peer)

    def _put(self, stream: Stream, peer: uuid.UUID) -> Future[None]:
        """Put stream into queue and notify"""
        future = super()._put(stream, peer)
        self._pool.submit(self._s2c, peer).add_done_callback(lambda future: future.result())
        return future

    def _s2c(self, peer: uuid.UUID):
        state = self._get_state(peer)

        state.put_flush()
        while not state.put_buffer.empty():
            with state.put_read(self._options.connection.max_size) as view:
                self._publish(f"s2c/{peer.hex}", bytes(view))

        if not state.state and state.put_empty():
            self._fin(peer)

    def _close(self) -> None:
        """Close the server"""

        # Wait peers to drain
        with self._lock:
            while self._peers:
                self._lock.wait()

        # Close resources
        self._pool.shutdown()
        self._stop_loop()

        # Unlock inflight external API:
        for _ in range(threading.active_count()):
            self._get_event.put(UUID_MAX)

        super()._close()
