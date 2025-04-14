"""Communications API test"""

import sys
import time
import enum
from pydtnn import comms
from argparse import ArgumentParser, Namespace


__all__ = ()


class Mode(enum.StrEnum):
    """Test modes"""
    SERVER = enum.auto()
    CLIENT = enum.auto()


# Argument pasrser
parser = ArgumentParser(prog="test_comms_api", description="Communications API test")
parser.add_argument("mode", choices=list(Mode))
parser.add_argument("--addr", type=str, default="127.0.0.1")
parser.add_argument("--port", type=int, default=50000)
parser.add_argument("--start-delay", type=float, default=3.0)
parser.add_argument("--size", type=int, default=1)


def server(config: Namespace):
    """Server mode"""
    messages = []

    with comms.Server(addr=config.addr, port=config.port) as server:

        for _ in range(config.size):
            client_msg = server.get()
            print(f"{server}-c2s: {client_msg}")
            messages.append(client_msg)

        server_msg = server._id
        print(f"{server}-s2c-global: {server_msg}")
        server.put(obj=server_msg)

        for server_msg in messages:
            print(f"{server}-s2c-local: {server_msg}")
            server.put(server_msg.obj, server_msg.peer)


def client(config: Namespace):
    """Client mode"""
    time.sleep(config.start_delay)
    with comms.Client(addr=config.addr, port=config.port) as client:

        put_msg = client._id
        print(f"{client}-c2s: {put_msg}")
        client.put(put_msg)

        get_msg = client.get()
        print(f"{client}-s2c-global: {get_msg}")
        assert client._server == get_msg.obj, "Corrupted message data"  # type: ignore

        get_msg = client.get()
        print(f"{client}-s2c-local: {get_msg}")
        assert put_msg == get_msg.obj, "Corrupted message data"


def main(config: Namespace):
    """Application entrypoint"""
    self = sys.modules[__name__]
    handler = getattr(self, config.mode)
    print(config)
    handler(config)


if __name__ == "__main__":
    main(parser.parse_args())
