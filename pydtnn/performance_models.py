""" Python Distributed Training of Neural Networks - PyDTNN

PyDTNN is a light-weight library for distributed Deep Learning training and 
inference that offers an initial starting point for interaction with 
distributed training of (and inference with) deep neural networks. PyDTNN 
priorizes simplicity over efficiency, providing an amiable user interface 
which enables a flat accessing curve. To perform the training and inference 
processes, PyDTNN exploits distributed inter-process parallelism (via MPI) 
for clusters and intra-process (via multi-threading) parallelism to leverage 
the presence of multicore processors and GPUs at node level. For that, PyDTNN 
uses MPI4Py for message-passing, BLAS calls via NumPy for multicore processors
and PyCUDA+cuDNN+cuBLAS for NVIDIA GPUs.

This program is free software: you can redistribute it and/or modify it under
the terms of the GNU General Public License as published by the Free Software
Foundation, either version 3 of the License, or (at your option) any later
version.

This program is distributed in the hope that it will be useful, but WITHOUT
ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
You should have received a copy of the GNU General Public License along with
this program. If not, see <http://www.gnu.org/licenses/>.

"""

__author__ = "Manuel F. Dolz, Enrique S. Quintana, \
              Mar Catalan, Adrian Castello"
__contact__ = "dolzm@uji.es"
__copyright__ = "Copyright 2020, Universitat Jaume I"
__credits__ = ["Manuel F. Dolz, Enrique S. Quintana", \
               "Mar Catalan", "Adrian Castello"]
__date__ = "2020/03/22"

__email__ = "dolzm@uji.es"
__license__ = "GPLv3"
__maintainer__ = "Manuel F. Dolz"
__status__ = "Production"
__version__ = "1.1.0"

from math import ceil, log
import numpy as np


def roofline(intensity, cpu_speed, memory_bw):
    # print ("COMPUTE_BOUND") if (cpu_speed < memory_bw * intens) else print ("MEMORY_BOUND")
    return min(cpu_speed, memory_bw * intensity)


def flops2time(flops, memops, cpu_speed, memory_bw, dtype):
    bfp = {np.float32: 4, np.float64: 8}[dtype]
    speed = roofline(flops / (bfp * memops), cpu_speed, memory_bw)
    time = flops / (speed + 1e-8)
    comp_time = flops / (cpu_speed + 1e-8)
    return np.array([time, comp_time, time - comp_time, 0], dtype=np.float32)


def im2col_time(m, n, cpu_speed, memory_bw, dtype):
    flops, memops = (0, m * n)
    return flops2time(flops, memops, cpu_speed, memory_bw, dtype)


def col2im_time(m, n, cpu_speed, memory_bw, dtype):
    flops, memops = (m * n, m * n)
    return flops2time(flops, memops, cpu_speed, memory_bw, dtype)


def matmul_time(m, n, k, cpu_speed, memory_bw, dtype):
    flops, memops = (2.0 * m * n * k, m * n + m * k + n * k)
    return flops2time(flops, memops, cpu_speed, memory_bw, dtype)


def allreduce_time(elems, cpu_speed, network_bw, network_lat, network_alg, nprocs, dtype):
    bfp = {np.float32: 4, np.float64: 8}[dtype]
    time = 0
    if network_alg == "bta":
        time = 2.0 * log(nprocs, 2) * network_lat + \
               2.0 * ((nprocs - 1.0) / nprocs) * ((elems * bfp * 8.0) / network_bw) + \
               ((nprocs - 1.0) / nprocs) * (elems / cpu_speed)
        comp_time = ((nprocs - 1.0) / nprocs) * (elems / cpu_speed)
    elif network_alg == "vdg":
        time = ceil(log(nprocs, 2)) * network_lat + \
               2.0 * ceil(log(nprocs, 2)) * ((elems * bfp * 8.0) / network_bw) + \
               ceil(log(nprocs, 2)) * (elems / cpu_speed)
        comp_time = ceil(log(nprocs, 2)) * (elems / cpu_speed)
    # print("allreduce_time; s; %8d; t; %8.8f" % (elems, time))
    return np.array([time, 0, 0, time], dtype=np.float32)


def scatter_time(elems, cpu_speed, network_bw, network_lat, network_alg, nprocs, dtype):
    bfp = {np.float32: 4, np.float64: 8}[dtype]
    time = 0
    if network_alg == "bta":
        time = ceil(log(nprocs, 2)) * network_lat + \
               ((nprocs - 1) / nprocs) * ((elems * bfp * 8.0) / network_bw)
    elif network_alg == "vdg":
        time = log(nprocs, 2) * network_lat + \
               ((nprocs - 1) / nprocs) * ((elems * bfp * 8.0) / network_bw)
    # print("scatter_time; s; %8d; t; %8.8f" % (elems, time))
    return np.array([time, 0, 0, time], dtype=np.float32)


def reduce_time(elems, cpu_speed, network_bw, network_lat, network_alg, nprocs, dtype):
    bfp = {np.float32: 4, np.float64: 8}[dtype]
    time, comp_time = 0, 0
    if network_alg == "bta":
        comp_time = ceil(log(nprocs, 2)) * (elems / cpu_speed)
        time = ceil(log(nprocs, 2)) * network_lat + \
               ceil(log(nprocs, 2)) * (elems * bfp * 8.0) / network_bw + \
               comp_time

    elif network_alg == "vdg":
        comp_time = ((nprocs - 1.0) / nprocs) * (elems / cpu_speed)
        time = 2.0 * log(nprocs, 2) * network_lat + \
               2.0 * ((nprocs - 1.0) / nprocs) * ((elems * bfp * 8.0) / network_bw) + \
               comp_time

    # print("reduce_time; s; %8d; t; %8.8f" % (elems, time))
    return np.array([time, comp_time, 0, time - comp_time], dtype=np.float32)


def bcast_time(elems, cpu_speed, network_bw, network_lat, network_alg, nprocs, dtype):
    bfp = {np.float32: 4, np.float64: 8}[dtype]
    time = 0
    if network_alg == "bta":
        time = ceil(log(nprocs, 2)) * ((3 * network_lat) + \
                                       ((elems * bfp * 8.0) / network_bw))
    elif network_alg == "vdg":
        time = (log(nprocs, 2) + nprocs - 1.0) * (network_lat) + \
               2.0 * ((nprocs - 1.0) / nprocs) * ((elems * bfp * 8.0) / network_bw)
    # print("bcast_time; s; %8d; t; %8.8f" % (elems, time))
    return np.array([time, 0, 0, time], dtype=np.float32)


def scatter_time(elems, cpu_speed, network_bw, network_lat, network_alg, nprocs, dtype):
    bfp = {np.float32: 4, np.float64: 8}[dtype]
    time = 0
    if network_alg == "bta":
        time = ceil(log(nprocs, 2)) * network_lat + \
               (((nprocs - 1) / nprocs)) * ((elems * bfp * 8.0) / network_bw)
    elif network_alg == "vdg":
        time = log(nprocs) * (network_lat) + \
               (((nprocs - 1) / nprocs)) * ((elems * bfp * 8.0) / network_bw)
    # print("scatter_time; s; %8d; t; %8.8f" % (elems, time))
    return time


def gather_time(elems, cpu_speed, network_bw, network_lat, network_alg, nprocs, dtype):
    time = bcast_time(elems, cpu_speed, network_bw, network_lat, network_alg, nprocs, dtype)
    # print("gather_time; s; %8d; t; %8.8f" % (elems, time))
    return time


def allgather_time(elems, cpu_speed, network_bw, network_lat, network_alg, nprocs, dtype):
    bfp = {np.float32: 4, np.float64: 8}[dtype]
    time = 0
    if network_alg == "bta":
        time = (nprocs - 1) * network_lat + \
               ((nprocs - 1) / nprocs) * ((elems * bfp * 8.0) / network_bw)
    elif network_alg == "vdg":
        time = ceil(log(nprocs, 2)) * (4 * network_lat) + \
               (((nprocs - 1) / nprocs) * ((elems * bfp * 8.0) / network_bw))
    # print("allgather_time; s; %8d; t; %8.8f" % (elems, time))
    return np.array([time, 0, 0, time], dtype=np.float32)


def reduce_scatter_time(elems, cpu_speed, network_bw, network_lat, network_alg, nprocs, dtype):
    bfp = {np.float32: 4, np.float64: 8}[dtype]
    time = 0
    if network_alg == "bta":
        comp_time = ((nprocs - 1) / nprocs) * (elems / cpu_speed)
        time = (nprocs - 1) * network_lat + \
               ((nprocs - 1) / nprocs) * ((elems * bfp * 8.0) / network_bw) + \
               comp_time
    elif network_alg == "vdg":
        comp_time = ((nprocs - 1) / nprocs) * (elems / cpu_speed)
        time = (nprocs - 1) * network_lat + \
               ((nprocs - 1) / nprocs) * ((elems * bfp * 8.0) / network_bw) + \
               comp_time
        # print("reduce_scatter_time; s; %8d; t; %8.8f" % (elems, time))
    return np.array([time, comp_time, 0, time - comp_time], dtype=np.float32)
