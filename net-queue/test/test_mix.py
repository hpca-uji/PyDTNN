"""Communications mix IOPS test"""

import sys
import time
import enum
import random
from pydtnn import utils
from threading import Thread
import net_queue as nq
from argparse import ArgumentParser, Namespace


__all__ = ()


class Peer(enum.StrEnum):
    """Peer type"""
    SERVER = enum.auto()
    CLIENT = enum.auto()


# Argument pasrser
parser = ArgumentParser(prog="test_comms_mix", description="Communications mix IOPS test")
parser.add_argument("proto", choices=list(nq.Protocol))
parser.add_argument("peer", choices=list(Peer))
parser.add_argument("--start-delay", type=float, default=3.0)
parser.add_argument("--delay", type=float, default=0.0)
parser.add_argument("--min_size", type=int, default=8)
parser.add_argument("--step-size", type=int, default=9)
parser.add_argument("--max-size", type=int, default=30)
parser.add_argument("--reps-expo", type=float, default=0.5)
parser.add_argument("--clients", type=int, default=1)


def get(comm: nq.Communicator, msgs: list[bytearray]):
    """Communication get handler"""
    for i in range(len(msgs)):
        print(i, end="\r", flush=True)
        comm.get().data
    print(i)


def put(comm: nq.Communicator, msgs: list[bytearray]):
    """Communication put handler"""
    for i in range(len(msgs)):
        print(i, end="\r", flush=True)
        comm.put(msgs[i])
    print(i)


def print_stats(config: Namespace, sizes: list[int], time: float) -> None:
    """Print statistics"""
    ops = len(sizes)
    size = sum(sizes)
    print(f"Time:       {time:.1f}s")
    print(f"Data:       {utils.convert_size(len(sizes))} @ {utils.convert_size(size)}B")
    print(f"Transfer:   {utils.convert_size(size)}B @ {utils.convert_size(size * 8 / time):>5}bps")  # type: ignore
    print(f"Operations: {utils.convert_size(ops)} @ {utils.convert_size(ops / time)}IOPS")  # type: ignore


def server(config: Namespace):
    """Server peer"""
    messages = []
    buffer = bytearray(2 ** config.max_size)
    for i in range(config.min_size, config.max_size, config.step_size):
        splits = round(((2 ** config.max_size) / (2 ** i)) ** config.reps_expo)
        for j in range(splits):
            messages.append(buffer[:2 ** i])
    messages.append(buffer)
    random.shuffle(messages)

    with nq.new(protocol=config.proto, purpose=nq.Purpose.SERVER) as server:
        get_thread = Thread(target=get, args=(server, messages * config.clients))
        put_thread = Thread(target=put, args=(server, messages))
        for _ in range(config.clients):
            server.get()
        server.put(None)
        start_time = time.time()

        get_thread.start()
        put_thread.start()
        get_thread.join()
        put_thread.join()

    end_time = time.time()

    print_stats(config=config, sizes=list(map(len, messages)) * config.clients, time=end_time - start_time)


def client(config: Namespace):
    """Client peer"""
    messages = []
    buffer = bytearray(2 ** config.max_size)
    for i in range(config.min_size, config.max_size, config.step_size):
        splits = round(((2 ** config.max_size) / (2 ** i)) ** config.reps_expo)
        for j in range(splits):
            messages.append(buffer[:2 ** i])
    messages.append(buffer)
    random.shuffle(messages)

    time.sleep(config.start_delay)
    with nq.new(protocol=config.proto, purpose=nq.Purpose.CLIENT) as client:
        get_thread = Thread(target=get, args=(client, messages))
        put_thread = Thread(target=put, args=(client, messages))
        client.put(None)
        client.get()
        start_time = time.time()

        get_thread.start()
        put_thread.start()
        get_thread.join()
        put_thread.join()

    end_time = time.time()

    print_stats(config=config, sizes=list(map(len, messages)), time=end_time - start_time)


def main(config: Namespace):
    """Application entrypoint"""
    self = sys.modules[__name__]
    handler = getattr(self, config.peer)
    print(config)
    random.seed(0)
    handler(config)


if __name__ == "__main__":
    main(parser.parse_args())
