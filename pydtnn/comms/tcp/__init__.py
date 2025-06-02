"""TCP communications"""

# FIXME: Handle SSL read/write error, handle SSL pendding

import socket
import selectors
from collections import abc
from queue import Empty, SimpleQueue
from concurrent.futures import ThreadPoolExecutor

from pydtnn import comms


__all__ = (
    "Protocol",
)


type Task = abc.Callable[[], None]


# Sentinel objects
CONTROL_STOP = object()
CONTROL_EVENT = b"\0"


class Protocol(comms.Communicator):
    """Shared base TCP implementation"""
    _max_message_size = 16 * 1024 ** 2 - 1
    _max_workers = 1

    def __init__(self, addr: str, port: int) -> None:
        """Inizialize comunicator"""
        super().__init__(addr, port)
        thread_prefix = f"{__name__}.{self.__class__.__qualname__}:{id(self)}"

        self._selector = selectors.DefaultSelector()
        self._control_socket = socket.socketpair()
        self._control_socket[1].setblocking(False)
        self._selector.register(self._control_socket[0], selectors.EVENT_READ, self._handle_control_socket)

        self._pool = ThreadPoolExecutor(max_workers=1 + self._max_workers, thread_name_prefix=f"{thread_prefix}")
        self._loop_thread = self._pool.submit(self._handle_selector_loop)
        # self._loop_thread.add_done_callback(lambda future: future.result())
        self._task_queue = SimpleQueue[Task]()

    def _modify_selector(self, fileobj, events) -> None:
        """Modify registered events"""
        try:
            key = self._selector.get_key(fileobj)
        except ValueError:
            return  # already removed

        def callback():
            try:
                self._selector.modify(key.fd, events, key.data)
            except KeyError:
                return  # already removed

        self._task_queue.put(callback)

    def _notify_selector(self) -> None:
        """Interrupt selector loop"""
        try:
            self._control_socket[1].send(CONTROL_EVENT)
        except BlockingIOError:
            pass  # already notified

    def _handle_control_socket(self, sock: socket.socket, mask):
        """Handle selector notification"""
        if len(sock.recv(self._max_message_size)) == 0:
            return CONTROL_STOP

        # Handle tasks
        while True:
            try:
                task = self._task_queue.get_nowait()
            except Empty:
                break
            else:
                task()

    def _handle_selector_event(self, event: tuple[selectors.SelectorKey, int]):
        """Handle selector event"""
        key, mask = event
        callback, fileobj = key.data, key.fileobj
        return callback(fileobj, mask)

    def _handle_selector_loop(self) -> None:
        """Handle selector loop"""
        running = True
        while running:
            for result in self._pool.map(self._handle_selector_event, self._selector.select()):
                if result is CONTROL_STOP:
                    running = False

    def _close(self) -> None:
        """Close the communication"""
        self._control_socket[1].close()
        self._loop_thread.result()
        self._control_socket[0].close()
        self._selector.close()
        self._pool.shutdown()
        super()._close()
