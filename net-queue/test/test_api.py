"""Communications API test"""

import sys
import time
import enum
import net_queue as nq
from argparse import ArgumentParser, Namespace


__all__ = ()


MSG = "Hello, World!"


class Mode(enum.StrEnum):
    """Test modes"""
    SERVER = enum.auto()
    CLIENT = enum.auto()


# Argument pasrser
parser = ArgumentParser(prog="test_comms_api", description="Communications API test")
parser.add_argument("proto", choices=list(nq.Protocol))
parser.add_argument("mode", choices=list(Mode))
parser.add_argument("--start-delay", type=float, default=3.0)
parser.add_argument("--size", type=int, default=1)


def server(config: Namespace):
    """Server mode"""
    clients = set()
    server_msg = MSG
    server = nq.new(protocol=config.proto, purpose=nq.Purpose.SERVER)

    for _ in range(config.size):
        client_msg = server.get()
        print(f"{server}-c2s: {client_msg}")
        clients.add(client_msg.peer)

    print(f"{server}-s2c-global: {server_msg}")
    future = server.put(data=server_msg)
    future.result()

    for client in clients:
        print(f"{server}-s2c-local: {server_msg}")
        server.put(server_msg, client)

    server.close()


def client(config: Namespace):
    """Client mode"""
    client_msg = MSG
    time.sleep(config.start_delay)
    client = nq.new(protocol=config.proto, purpose=nq.Purpose.CLIENT)

    print(f"{client}-c2s: {client_msg}")
    future = client.put(client_msg)
    future.result()

    server_msg = client.get()
    print(f"{client}-s2c-global: {server_msg}")
    assert client_msg == server_msg.data, "Corrupted message data"  # type: ignore

    server_msg = client.get()
    print(f"{client}-s2c-local: {server_msg}")
    assert client_msg == server_msg.data, "Corrupted message data"

    client.close()


def main(config: Namespace):
    """Application entrypoint"""
    self = sys.modules[__name__]
    handler = getattr(self, config.mode)
    print(config)
    handler(config)


if __name__ == "__main__":
    main(parser.parse_args())
