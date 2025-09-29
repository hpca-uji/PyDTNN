"""Communications IOPS test"""

import sys
import time
import enum
import numpy
from threading import Thread
from pydtnn import utils
import net_queue as nq
from argparse import ArgumentParser, Namespace


__all__ = ()


class Peer(enum.StrEnum):
    """Peer type"""
    SERVER = enum.auto()
    CLIENT = enum.auto()


class Mode(enum.StrEnum):
    """Test mode"""
    SEQUENTIAL = enum.auto()
    RANDOM = enum.auto()


# Argument pasrser
parser = ArgumentParser(prog="test_comms_iops", description="Communications IOPS test")
parser.add_argument("proto", choices=list(nq.Protocol))
parser.add_argument("peer", choices=list(Peer))
parser.add_argument("mode", choices=list(Mode))
parser.add_argument("--start-delay", type=float, default=3.0)
parser.add_argument("--delay", type=float, default=0.0)
parser.add_argument("--size", type=int, default=1_000)
parser.add_argument("--reps", type=int, default=1_000_000)
parser.add_argument("--clients", type=int, default=1)


def get(comm: nq.Communicator, msg: numpy.ndarray, reps: int):
    """Communication get handler"""
    for i in range(reps):
        print(i, end="\r", flush=True)
        got = comm.get().data
        assert len(got) == len(msg), f"Lost message data (got {len(got)}, expect {len(msg)})"
    assert numpy.array_equal(got, msg), "Corrupted message data"
    print(i)
    return got


def put(comm: nq.Communicator, msg: numpy.ndarray, reps: int):
    """Communication put handler"""
    for i in range(reps):
        print(i, end="\r", flush=True)
        comm.put(msg)
    print(i)
    return msg


def print_stats(config: Namespace, time: float) -> None:
    """Print statistics"""
    ops = config.reps * 2
    size = config.size * ops
    print(f"Time:       {time:.1f}s")
    print(f"Data:       {utils.convert_size(config.reps)} x {utils.convert_size(config.size)}B")
    print(f"Transfer:   {utils.convert_size(size)}B @ {utils.convert_size(size * 8 / time):>5}bps")
    print(f"Operations: {utils.convert_size(ops)} @ {utils.convert_size(ops / time)}IOPS")


def server(config: Namespace):
    """Server peer"""
    message = numpy.arange(config.size, dtype=numpy.uint8)

    with nq.new(protocol=config.proto, purpose=nq.Purpose.SERVER) as server:
        get_thread = Thread(target=get, args=(server, message, config.reps * config.clients))
        put_thread = Thread(target=put, args=(server, message, config.reps))
        for _ in range(config.clients):
            server.get()
        server.put(None)
        start_time = time.time()

        match config.mode:
            case Mode.SEQUENTIAL:
                time.sleep(config.delay)
                get_thread.run()
                put_thread.run()

            case Mode.RANDOM:
                get_thread.start()
                put_thread.start()
                get_thread.join()
                put_thread.join()

    end_time = time.time()
    del message

    print_stats(config=config, time=end_time - start_time)


def client(config: Namespace):
    """Client peer"""
    message = numpy.arange(config.size, dtype=numpy.uint8)

    time.sleep(config.start_delay)
    with nq.new(protocol=config.proto, purpose=nq.Purpose.CLIENT) as client:
        get_thread = Thread(target=get, args=(client, message, config.reps))
        put_thread = Thread(target=put, args=(client, message, config.reps))
        client.put(None)
        client.get()
        start_time = time.time()

        match config.mode:
            case Mode.SEQUENTIAL:
                put_thread.run()
                time.sleep(config.delay)
                get_thread.run()

            case Mode.RANDOM:
                get_thread.start()
                put_thread.start()
                get_thread.join()
                put_thread.join()

    end_time = time.time()
    del message

    print_stats(config=config, time=end_time - start_time)


def main(config: Namespace):
    """Application entrypoint"""
    self = sys.modules[__name__]
    handler = getattr(self, config.peer)
    print(config)
    handler(config)


if __name__ == "__main__":
    main(parser.parse_args())
