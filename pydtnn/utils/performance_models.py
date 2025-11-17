from math import ceil, log
import numpy as np
from pydtnn.utils.constants import NetworkAlgEnum


def roofline(intensity, cpu_speed, memory_bw):
    # print ("COMPUTE_BOUND") if (cpu_speed < memory_bw * intens) else print ("MEMORY_BOUND")
    return min(cpu_speed, memory_bw * intensity)


def flops2time(flops: int, memops: int, cpu_speed: float, memory_bw: float, dtype: type | np.dtype) -> np.ndarray[np.float32]:
    bfp = np.dtype(dtype).itemsize
    speed = roofline(flops / (bfp * memops), cpu_speed, memory_bw)
    time = flops / (speed + 1e-8)
    comp_time = flops / (cpu_speed + 1e-8)
    return np.array([time, comp_time, time - comp_time, 0], dtype=np.float32)


def im2col_time(m: int, n: int, cpu_speed: float, memory_bw: float, dtype: type | np.dtype) -> np.ndarray[np.float32]:
    flops, memops = (0, m * n)
    return flops2time(flops, memops, cpu_speed, memory_bw, dtype)


def col2im_time(m: int, n: int, cpu_speed: float, memory_bw: float, dtype: type | np.dtype) -> np.ndarray[np.float32]:
    flops, memops = (m * n, m * n)
    return flops2time(flops, memops, cpu_speed, memory_bw, dtype)


def matmul_time(m: int, n: int, k: int, cpu_speed: float, memory_bw: float, dtype: type | np.dtype) -> np.ndarray[np.float32]:
    flops, memops = (2.0 * m * n * k, m * n + m * k + n * k)
    return flops2time(flops, memops, cpu_speed, memory_bw, dtype)


def allreduce_time(elems: int, cpu_speed: float, network_bw: float, network_lat: float,
                   network_alg: str, nprocs: int, dtype: type | np.dtype) -> np.ndarray[np.float32]:
    bfp = np.dtype(dtype).itemsize
    time = 0
    match network_alg:
        case NetworkAlgEnum.BTA:
            time = 2.0 * log(nprocs, 2) * network_lat + \
                2.0 * ((nprocs - 1.0) / nprocs) * ((elems * bfp * 8.0) / network_bw) + \
                ((nprocs - 1.0) / nprocs) * (elems / cpu_speed)
            comp_time = ((nprocs - 1.0) / nprocs) * (elems / cpu_speed)
        case NetworkAlgEnum.VDG:
            time = ceil(log(nprocs, 2)) * network_lat + \
                2.0 * ceil(log(nprocs, 2)) * ((elems * bfp * 8.0) / network_bw) + \
                ceil(log(nprocs, 2)) * (elems / cpu_speed)
            comp_time = ceil(log(nprocs, 2)) * (elems / cpu_speed)
        case _ :
            raise ValueError(f"network_alg ({network_alg}) not in {list(NetworkAlgEnum)}")
    # print("allreduce_time; s; %8d; t; %8.8f" % (elems, time))
    return np.array([time, 0, 0, time], dtype=np.float32)


def scatter_time(elems: int, cpu_speed: float, network_bw: float, network_lat: float,
                 network_alg: str, nprocs: int, dtype: type | np.dtype) -> np.ndarray[np.float32]:
    bfp = np.dtype(dtype).itemsize
    time = 0
    match network_alg:
        case NetworkAlgEnum.BTA:
            time = ceil(log(nprocs, 2)) * network_lat + \
                ((nprocs - 1) / nprocs) * ((elems * bfp * 8.0) / network_bw)
        case NetworkAlgEnum.VDG:
            time = log(nprocs, 2) * network_lat + \
                ((nprocs - 1) / nprocs) * ((elems * bfp * 8.0) / network_bw)
        case _ :
            raise ValueError(f"network_alg ({network_alg}) not in {list(NetworkAlgEnum)}")
    # print("scatter_time; s; %8d; t; %8.8f" % (elems, time))
    return np.array([time, 0, 0, time], dtype=np.float32)


def reduce_time(elems: int, cpu_speed: float, network_bw: float, network_lat: float,
                network_alg: str, nprocs: int, dtype: type | np.dtype) -> np.ndarray[np.float32]:
    bfp = np.dtype(dtype).itemsize
    time, comp_time = 0, 0
    match network_alg:
        case NetworkAlgEnum.BTA:
            comp_time = ceil(log(nprocs, 2)) * (elems / cpu_speed)
            time = ceil(log(nprocs, 2)) * network_lat + \
                ceil(log(nprocs, 2)) * (elems * bfp * 8.0) / network_bw + \
                comp_time
        case NetworkAlgEnum.VDG:
            comp_time = ((nprocs - 1.0) / nprocs) * (elems / cpu_speed)
            time = 2.0 * log(nprocs, 2) * network_lat + \
                2.0 * ((nprocs - 1.0) / nprocs) * ((elems * bfp * 8.0) / network_bw) + \
                comp_time        
        case _ :
            raise ValueError(f"network_alg ({network_alg}) not in {list(NetworkAlgEnum)}")

    # print("reduce_time; s; %8d; t; %8.8f" % (elems, time))
    return np.array([time, comp_time, 0, time - comp_time], dtype=np.float32)


def bcast_time(elems: int, cpu_speed: float, network_bw: float, network_lat: float,
               network_alg: str, nprocs: int, dtype: type | np.dtype) -> np.ndarray[np.float32]:
    bfp = np.dtype(dtype).itemsize
    time = 0
    match network_alg:
        case NetworkAlgEnum.BTA:
            time = ceil(log(nprocs, 2)) * ((3 * network_lat) +
                                        ((elems * bfp * 8.0) / network_bw))
        case NetworkAlgEnum.VDG:
            time = (log(nprocs, 2) + nprocs - 1.0) * (network_lat) + \
                2.0 * ((nprocs - 1.0) / nprocs) * ((elems * bfp * 8.0) / network_bw)
        # print("bcast_time; s; %8d; t; %8.8f" % (elems, time))
        case _ :
            raise ValueError(f"network_alg ({network_alg}) not in {list(NetworkAlgEnum)}")

    return np.array([time, 0, 0, time], dtype=np.float32)


def scatter_time(elems: int, cpu_speed: float, network_bw: float, network_lat: float,
                 network_alg: str, nprocs: int, dtype: type | np.dtype) -> np.ndarray[np.float32]:
    bfp = np.dtype(dtype).itemsize
    time = 0
    match network_alg:
        case NetworkAlgEnum.BTA:
            time = ceil(log(nprocs, 2)) * network_lat + \
                (((nprocs - 1) / nprocs)) * ((elems * bfp * 8.0) / network_bw)
        case NetworkAlgEnum.VDG:
            time = log(nprocs) * (network_lat) + \
                (((nprocs - 1) / nprocs)) * ((elems * bfp * 8.0) / network_bw)
        case _ :
            raise ValueError(f"network_alg ({network_alg}) not in {list(NetworkAlgEnum)}")
    # print("scatter_time; s; %8d; t; %8.8f" % (elems, time))
    return time


def gather_time(elems: int, cpu_speed: float, network_bw: float, network_lat: float,
                network_alg: str, nprocs: int, dtype: type | np.dtype) -> np.ndarray[np.float32]:
    time = bcast_time(elems, cpu_speed, network_bw, network_lat, network_alg, nprocs, dtype)
    # print("gather_time; s; %8d; t; %8.8f" % (elems, time))
    return time


def allgather_time(elems: int, cpu_speed: float, network_bw: float, network_lat: float,
                   network_alg: str, nprocs: int, dtype: type | np.dtype) -> np.ndarray[np.float32]:
    bfp = np.dtype(dtype).itemsize
    time = 0
    match network_alg:
        case NetworkAlgEnum.BTA:
            time = (nprocs - 1) * network_lat + \
                ((nprocs - 1) / nprocs) * ((elems * bfp * 8.0) / network_bw)
        case NetworkAlgEnum.VDG:
            time = ceil(log(nprocs, 2)) * (4 * network_lat) + \
                (((nprocs - 1) / nprocs) * ((elems * bfp * 8.0) / network_bw))
        case _ :
            raise ValueError(f"network_alg ({network_alg}) not in {list(NetworkAlgEnum)}")
    # print("allgather_time; s; %8d; t; %8.8f" % (elems, time))
    return np.array([time, 0, 0, time], dtype=np.float32)


def reduce_scatter_time(elems: int, cpu_speed: float, network_bw: float, network_lat: float,
                        network_alg: str, nprocs: int, dtype: type | np.dtype) -> np.ndarray[np.float32]:
    bfp = np.dtype(dtype).itemsize
    time = 0
    match network_alg:
        case NetworkAlgEnum.BTA:
            comp_time = ((nprocs - 1) / nprocs) * (elems / cpu_speed)
            time = (nprocs - 1) * network_lat + \
                ((nprocs - 1) / nprocs) * ((elems * bfp * 8.0) / network_bw) + \
                comp_time
        case NetworkAlgEnum.VDG:
            comp_time = ((nprocs - 1) / nprocs) * (elems / cpu_speed)
            time = (nprocs - 1) * network_lat + \
                ((nprocs - 1) / nprocs) * ((elems * bfp * 8.0) / network_bw) + \
                comp_time
        case _ :
            raise ValueError(f"network_alg ({network_alg}) not in {list(NetworkAlgEnum)}")
        # print("reduce_scatter_time; s; %8d; t; %8.8f" % (elems, time))
    return np.array([time, comp_time, 0, time - comp_time], dtype=np.float32)
