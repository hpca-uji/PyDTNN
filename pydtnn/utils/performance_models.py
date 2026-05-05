import logging
from math import ceil, log

import numpy as np

from pydtnn.utils.constants import NetworkAlgoEnum

__all__ = (
    "allgather_time",
    "allreduce_time",
    "bcast_time",
    "col2im_time",
    "flops2time",
    "gather_time",
    "im2col_time",
    "matmul_time",
    "reduce_scatter_time",
    "reduce_time",
    "roofline",
    "scatter_time",
    "scatter_time",
)

logger = logging.getLogger(__name__)


def roofline(intensity, cpu_speed, memory_bw):
    # print ("COMPUTE_BOUND") if (cpu_speed < memory_bw * intens) else print ("MEMORY_BOUND")
    return min(cpu_speed, memory_bw * intensity)


def flops2time(flops: float, memops: int, cpu_speed: float, memory_bw: float, dtype: type | np.dtype) -> np.ndarray[tuple[int, int, int, int], np.dtype[np.float32]]:
    bfp = np.dtype(dtype).itemsize
    speed = roofline(flops / (bfp * memops), cpu_speed, memory_bw)
    time = flops / (speed + 1e-8)
    comp_time = flops / (cpu_speed + 1e-8)
    return np.array([time, comp_time, time - comp_time, 0.0], dtype=np.float32)


def im2col_time(m: int, n: int, cpu_speed: float, memory_bw: float, dtype: type | np.dtype) -> np.ndarray[tuple[int, int, int, int], np.dtype[np.float32]]:
    flops, memops = (0, m * n)
    return flops2time(flops, memops, cpu_speed, memory_bw, dtype)


def col2im_time(m: int, n: int, cpu_speed: float, memory_bw: float, dtype: type | np.dtype) -> np.ndarray[tuple[int, int, int, int], np.dtype[np.float32]]:
    flops, memops = (m * n, m * n)
    return flops2time(flops, memops, cpu_speed, memory_bw, dtype)


def matmul_time(m: int, n: int, k: int, cpu_speed: float, memory_bw: float, dtype: type | np.dtype) -> np.ndarray[tuple[int, int, int, int], np.dtype[np.float32]]:
    flops, memops = (2.0 * m * n * k, m * n + m * k + n * k)
    return flops2time(flops, memops, cpu_speed, memory_bw, dtype)


def allreduce_time(elems: int, cpu_speed: float, network_bw: float, network_lat: float,
                   network_algo: str, nprocs: int, dtype: type | np.dtype) -> np.ndarray[tuple[int, int, int, int], np.dtype[np.float32]]:
    bfp = np.dtype(dtype).itemsize
    time = 0
    match network_algo:
        case NetworkAlgoEnum.BTA:
            time = 2.0 * log(nprocs, 2) * network_lat + \
                2.0 * ((nprocs - 1.0) / nprocs) * ((elems * bfp * 8.0) / network_bw) + \
                ((nprocs - 1.0) / nprocs) * (elems / cpu_speed)
            comp_time = ((nprocs - 1.0) / nprocs) * (elems / cpu_speed)
        case NetworkAlgoEnum.VDG:
            time = ceil(log(nprocs, 2)) * network_lat + \
                2.0 * ceil(log(nprocs, 2)) * ((elems * bfp * 8.0) / network_bw) + \
                ceil(log(nprocs, 2)) * (elems / cpu_speed)
            comp_time = ceil(log(nprocs, 2)) * (elems / cpu_speed)
        case _:
            raise ValueError(f"network_algo ({network_algo}) not in {list(NetworkAlgoEnum)}")
    # print("allreduce_time; s; %8d; t; %8.8f" % (elems, time))
    return np.array([time, 0, 0, time], dtype=np.float32)


def scatter_time(  # type: ignore (override)
        elems: int, cpu_speed: float, network_bw: float, network_lat: float,
        network_algo: str, nprocs: int, dtype: type | np.dtype) -> np.ndarray[tuple[int, int, int, int], np.dtype[np.float32]]:
    bfp = np.dtype(dtype).itemsize
    time = 0
    match network_algo:
        case NetworkAlgoEnum.BTA:
            time = ceil(log(nprocs, 2)) * network_lat + \
                ((nprocs - 1) / nprocs) * ((elems * bfp * 8.0) / network_bw)
        case NetworkAlgoEnum.VDG:
            time = log(nprocs, 2) * network_lat + \
                ((nprocs - 1) / nprocs) * ((elems * bfp * 8.0) / network_bw)
        case _:
            raise ValueError(f"network_algo ({network_algo}) not in {list(NetworkAlgoEnum)}")
    # print("scatter_time; s; %8d; t; %8.8f" % (elems, time))
    return np.array([time, 0, 0, time], dtype=np.float32)


def reduce_time(elems: int, cpu_speed: float, network_bw: float, network_lat: float,
                network_algo: str, nprocs: int, dtype: type | np.dtype) -> np.ndarray[tuple[int, int, int, int], np.dtype[np.float32]]:
    bfp = np.dtype(dtype).itemsize
    time, comp_time = 0, 0
    match network_algo:
        case NetworkAlgoEnum.BTA:
            comp_time = ceil(log(nprocs, 2)) * (elems / cpu_speed)
            time = ceil(log(nprocs, 2)) * network_lat + \
                ceil(log(nprocs, 2)) * (elems * bfp * 8.0) / network_bw + \
                comp_time
        case NetworkAlgoEnum.VDG:
            comp_time = ((nprocs - 1.0) / nprocs) * (elems / cpu_speed)
            time = 2.0 * log(nprocs, 2) * network_lat + \
                2.0 * ((nprocs - 1.0) / nprocs) * ((elems * bfp * 8.0) / network_bw) + \
                comp_time
        case _:
            raise ValueError(f"network_algo ({network_algo}) not in {list(NetworkAlgoEnum)}")

    # print("reduce_time; s; %8d; t; %8.8f" % (elems, time))
    return np.array([time, comp_time, 0, time - comp_time], dtype=np.float32)


def bcast_time(elems: int, cpu_speed: float, network_bw: float, network_lat: float,
               network_algo: str, nprocs: int, dtype: type | np.dtype) -> np.ndarray[tuple[int, int, int, int], np.dtype[np.float32]]:
    bfp = np.dtype(dtype).itemsize
    time = 0
    match network_algo:
        case NetworkAlgoEnum.BTA:
            time = ceil(log(nprocs, 2)) * ((3 * network_lat) +
                                           ((elems * bfp * 8.0) / network_bw))
        case NetworkAlgoEnum.VDG:
            time = (log(nprocs, 2) + nprocs - 1.0) * (network_lat) + \
                2.0 * ((nprocs - 1.0) / nprocs) * ((elems * bfp * 8.0) / network_bw)
        # print("bcast_time; s; %8d; t; %8.8f" % (elems, time))
        case _:
            raise ValueError(f"network_algo ({network_algo}) not in {list(NetworkAlgoEnum)}")

    return np.array([time, 0, 0, time], dtype=np.float32)


def scatter_time(elems: int, cpu_speed: float, network_bw: float, network_lat: float,
                 network_algo: str, nprocs: int, dtype: type | np.dtype) -> np.ndarray[tuple[int, int, int, int], np.dtype[np.float32]]:
    bfp = np.dtype(dtype).itemsize
    time = 0
    match network_algo:
        case NetworkAlgoEnum.BTA:
            time = ceil(log(nprocs, 2)) * network_lat + \
                (((nprocs - 1) / nprocs)) * ((elems * bfp * 8.0) / network_bw)
        case NetworkAlgoEnum.VDG:
            time = log(nprocs) * (network_lat) + \
                (((nprocs - 1) / nprocs)) * ((elems * bfp * 8.0) / network_bw)
        case _:
            raise ValueError(f"network_algo ({network_algo}) not in {list(NetworkAlgoEnum)}")
    # print("scatter_time; s; %8d; t; %8.8f" % (elems, time))
    return np.array([time, 0, 0, time], dtype=np.float32)


def gather_time(elems: int, cpu_speed: float, network_bw: float, network_lat: float,
                network_algo: str, nprocs: int, dtype: type | np.dtype) -> np.ndarray[tuple[int, int, int, int], np.dtype[np.float32]]:
    time = bcast_time(elems, cpu_speed, network_bw, network_lat, network_algo, nprocs, dtype)
    # print("gather_time; s; %8d; t; %8.8f" % (elems, time))
    return time


def allgather_time(elems: int, cpu_speed: float, network_bw: float, network_lat: float,
                   network_algo: str, nprocs: int, dtype: type | np.dtype) -> np.ndarray[tuple[int, int, int, int], np.dtype[np.float32]]:
    bfp = np.dtype(dtype).itemsize
    time = 0
    match network_algo:
        case NetworkAlgoEnum.BTA:
            time = (nprocs - 1) * network_lat + \
                ((nprocs - 1) / nprocs) * ((elems * bfp * 8.0) / network_bw)
        case NetworkAlgoEnum.VDG:
            time = ceil(log(nprocs, 2)) * (4 * network_lat) + \
                (((nprocs - 1) / nprocs) * ((elems * bfp * 8.0) / network_bw))
        case _:
            raise ValueError(f"network_algo ({network_algo}) not in {list(NetworkAlgoEnum)}")
    # print("allgather_time; s; %8d; t; %8.8f" % (elems, time))
    return np.array([time, 0, 0, time], dtype=np.float32)


def reduce_scatter_time(elems: int, cpu_speed: float, network_bw: float, network_lat: float,
                        network_algo: str, nprocs: int, dtype: type | np.dtype) -> np.ndarray[tuple[int, int, int, int], np.dtype[np.float32]]:
    bfp = np.dtype(dtype).itemsize
    time = 0
    match network_algo:
        case NetworkAlgoEnum.BTA:
            comp_time = ((nprocs - 1) / nprocs) * (elems / cpu_speed)
            time = (nprocs - 1) * network_lat + \
                ((nprocs - 1) / nprocs) * ((elems * bfp * 8.0) / network_bw) + \
                comp_time
        case NetworkAlgoEnum.VDG:
            comp_time = ((nprocs - 1) / nprocs) * (elems / cpu_speed)
            time = (nprocs - 1) * network_lat + \
                ((nprocs - 1) / nprocs) * ((elems * bfp * 8.0) / network_bw) + \
                comp_time
        case _:
            raise ValueError(f"network_algo ({network_algo}) not in {list(NetworkAlgoEnum)}")
        # print("reduce_scatter_time; s; %8d; t; %8.8f" % (elems, time))
    return np.array([time, comp_time, 0, time - comp_time], dtype=np.float32)
