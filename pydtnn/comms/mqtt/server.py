"""MQTT server"""

import uuid
import threading
from concurrent.futures import Future

import paho.mqtt.client as mqtt_client

from pydtnn.comms import server
from pydtnn.comms.mqtt import Protocol
from pydtnn.utils.io_stream import Stream
from pydtnn.utils import UUID_MAX, UUID_NIL
from pydtnn.utils.asynctools import merge_futures
from pydtnn.comms import CommunicatorOptions, ResourceClosed, Message


__all__ = (
    "Server",
)


# Sentinel objects
END_COMM = b""


class Server(Protocol, server.Server):
    """MQTT server"""

    def __init__(self, options: CommunicatorOptions = CommunicatorOptions()) -> None:
        """Server initialization"""
        super().__init__(options)

        # MQTT
        self._register_handler(topic="c2s/+", handler=self._c2s)

    def _new_connection(self, sock: str) -> uuid.UUID:
        """Handle new incomming connections"""
        # NOTE: communication thead
        peer = uuid.UUID(hex=sock)

        with self._lock:
            self._peers[peer] = sock
            self._state[peer] = self._new_state()

            # ACK
            self._put(self._session_ini(peer), peer)
            self._lock.notify_all()

        return peer

    def _c2s(self, client: mqtt_client.Client, userdata, mqtt_message: mqtt_client.MQTTMessage) -> None:
        """Client message handler"""
        # NOTE: communication thead
        sock = self._peer(mqtt_message)
        try:
            peer = self._peers.inverse[sock]
        except KeyError:
            peer = self._new_connection(sock)
        state = self._state[peer]

        data = mqtt_message.payload
        state.get_write(data)
        peer = self._get_flush(peer)

    def _fin(self, peer: uuid.UUID) -> None:
        """Close connection"""

        # Remove peer
        with self._lock:
            del self._peers[peer]

            # TODO: reuse peer_cleanup
            if self._state[peer].get_empty():
                del self._state[peer]

            self._lock.notify_all()

    def _get_flush(self, peer: uuid.UUID) -> uuid.UUID:
        state = self._state[peer]

        for stream in state.get_flush():
            if stream.empty():
                self._handle_session_fin(peer, stream)
                self._put(self._session_fin(peer), peer)

            elif state.peer == UUID_NIL:
                self._handle_session_ini(peer, stream)
                peer = state.peer

            else:
                state.get_queue.put(stream)
                self._get_event.put(peer)

        return peer

    def get(self, *peers: uuid.UUID) -> Message:
        """Get data from a client"""
        # NOTE: peers could be missing or disconnect creating infinite wait, which is an expected state during startup
        assert len(peers) == 0, "Server can not get from specific client"
        peer = self._get_event.get()

        # Exit signaled
        if peer == UUID_MAX:
            raise ResourceClosed()

        state = self._state[peer]
        get_queue = state.get_queue

        # Get object
        stream = get_queue.get_nowait()

        self._peer_cleanup(peer)

        with stream:
            obj = self._serializer.load(stream)

        return Message(peer=peer, obj=obj)

    def _put(self, stream: Stream, peer: uuid.UUID) -> Future[None]:
        """Put stream into queue and notify"""
        try:
            state = self._state[peer]
            future = state.put(stream)
        except (KeyError, ResourceClosed):
            raise ResourceClosed(peer)
        self._pool.submit(self._s2c, peer).add_done_callback(lambda future: future.result())
        return future

    def _s2c(self, peer: uuid.UUID):
        state = self._state[peer]

        state.put_flush()
        while not state.put_buffer.empty():
            with state.put_read(self._options.connection.max_size) as view:
                self._publish(f"s2c/{peer.hex}", bytes(view))

        if not state.state and state.put_empty():
            self._fin(peer)

    def put(self, obj, *peers: uuid.UUID) -> Future[None]:
        """Publish data to clients"""
        if not peers:
            with self._lock:
                peers = tuple(self._peers)

        futures = list[Future[None]]()
        errors = list[ResourceClosed]()
        with self._serializer.dump(obj) as stream:
            for peer in peers:
                try:
                    future = self._put(stream.copy(), peer)
                except ResourceClosed as exc:
                    errors.append(exc)
                    continue
                else:
                    futures.append(future)

        if errors:
            raise ExceptionGroup("Peer does not exist", errors)

        return merge_futures(futures)

    def _close(self) -> None:
        """Close the server"""

        # Wait peers to drain
        with self._lock:
            while self._peers:
                self._lock.wait()

        # Unlock inflight external API:
        for _ in range(threading.active_count()):
            self._get_event.put(UUID_MAX)

        # Close resources
        self._pool.shutdown()
        self._stop_loop()
        super()._close()
