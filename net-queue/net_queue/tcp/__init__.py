"""TCP communications"""

import socket
import warnings
import selectors
from collections import abc
from concurrent import futures
from queue import Empty, SimpleQueue

import net_queue as nq
from net_queue import asynctools
from net_queue.asynctools import thread_func


__all__ = (
    "Protocol",
)


type Task = abc.Callable[[], None]


# Sentinel objects
CONTROL_STOP = object()
CONTROL_EVENT = b"\0"


class Protocol[T](nq.Communicator[T]):
    """Shared base TCP implementation"""

    def __init__(self, options: nq.CommunicatorOptions = nq.CommunicatorOptions()) -> None:
        """Inizialize comunicator"""
        super().__init__(options)

        self._selector = selectors.DefaultSelector()
        self._control_socket = socket.socketpair()
        self._control_socket[1].setblocking(False)
        self._selector.register(self._control_socket[0], selectors.EVENT_READ, self._handle_control_socket)

        self._loop_thread = thread_func(self._handle_selector_loop)
        self._loop_thread.add_done_callback(asynctools.future_warn_exception)
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
        if len(sock.recv(self.options.connection.max_size)) == 0:
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
            fs = [
                self._pool.submit(self._handle_selector_event, event)
                for event in self._selector.select()
            ]

            for future in futures.as_completed(fs):
                try:
                    result = future.result()
                except Exception as exc:
                    warnings.warn(repr(exc), RuntimeWarning)
                else:
                    if result is CONTROL_STOP:
                        running = False

    def _close(self) -> None:
        """Close the communication"""
        self._control_socket[1].close()
        self._loop_thread.result()
        self._control_socket[0].close()
        self._selector.close()
        super()._close()
