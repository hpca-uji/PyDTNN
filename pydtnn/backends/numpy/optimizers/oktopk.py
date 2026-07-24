"""Module for the OkTopk optimizer implementation using NumPy."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from pydtnn.backends.numpy.abstract.layerable import Layerable
from pydtnn.backends.numpy.optimizers.abstract.optimizer import OptimizerNumpy
from pydtnn.libs import numpy as np
from pydtnn.optimizers.oktopk import OkTopk
from pydtnn.utils.sparse import SparseFlatArray

__all__ = ("OkTopkNumpy",)

logger = logging.getLogger(__name__)

type BoundaryArray = np.ndarray[tuple[int], np.dtype[np.int32]]


if TYPE_CHECKING:
    import numpy as np  # noqa: F811 (override typing)
    from pympi.MPI import Request


try:
    from pydtnn.libs.mpi import MPI
except (ImportError, ModuleNotFoundError):
    MPI = None


class OkTopkNumpy(OkTopk[np.ndarray], OptimizerNumpy):
    """NumPy-based implementation of the OkTopk optimizer."""

    def _model_init(self, layers: list[Layerable]) -> None:
        """
        Initializes model-specific structures for the optimizer.

        Args:
            layers: List of layers to be optimized.
        """
        super()._model_init(layers)

        self.iterations: dict[int, int] = {}
        self.all_local_th: dict[int, dict[str, float]] = {}
        self.all_global_th: dict[int, dict[str, float]] = {}
        self.all_residuals: dict[int, dict[str, np.ndarray]] = {}
        self.all_velocity: dict[int, dict[str, np.ndarray]] = {}
        self.all_boundaries: dict[int, dict[str, BoundaryArray]] = {}

        for layer in self.layers:
            self.iterations[layer.id] = 0

            self.all_local_th[layer.id] = {dw_: 0.0 for dw_ in layer.grad_vars.values()}
            self.all_global_th[layer.id] = {dw_: 0.0 for dw_ in layer.grad_vars.values()}

            self.all_residuals[layer.id] = {
                dw_: np.zeros_like(getattr(layer, dw_)) for dw_ in layer.grad_vars.values()
            }

            self.all_velocity[layer.id] = {
                dw_: np.zeros_like(getattr(layer, dw_)) for dw_ in layer.grad_vars.values()
            }

            self.all_boundaries[layer.id] = {
                dw_: np.zeros(self.model.comm_size, dtype=np.int32)
                for dw_ in layer.grad_vars.values()
            }

            for dw_ in layer.grad_vars.values():
                self.memory_used += self.all_residuals[layer.id][dw_].nbytes
                self.memory_used += self.all_velocity[layer.id][dw_].nbytes
                self.memory_used += self.all_boundaries[layer.id][dw_].nbytes

        self._select_methods()

    def _select_methods(self, comm: bool = True) -> None:
        """Select method alternatives"""
        # Partition
        if comm and self.model.comm_size > 1:
            match self.model.oktopk_partition_method:
                case "dense":
                    self._space_repartition = self._space_repartition_dense
                case "sparse":
                    self._space_repartition = self._space_repartition_sparse
                case _:
                    raise NotImplementedError(
                        f"Method '{self.model.oktopk_partition_method}' not implemented"
                    )
        else:
            self._space_repartition = self._space_repartition_local

        # Reduce
        if comm and self.model.comm_size > 1:
            match self.model.oktopk_reduce_method:
                case "collective_allreduce_then_slice":
                    self._reduce_topk = self._reduce_topk_collective_allreduce_then_slice
                case "collective_region_wise_reduce_sync":
                    self._reduce_topk = self._reduce_topk_collective_region_wise_reduce_sync
                case "collective_region_wise_reduce_async":
                    self._reduce_topk = self._reduce_topk_collective_region_wise_reduce_async
                case "p2p_region_wise_reduce_static_destination":
                    self._reduce_topk = self._reduce_topk_p2p_region_wise_reduce_static_destination
                case "p2p_region_wise_reduce_destination_rotation_and_bucketing":
                    self._reduce_topk = (
                        self._reduce_topk_p2p_region_wise_reduce_destination_rotation_and_bucketing
                    )
                case _:
                    raise NotImplementedError(
                        f"Method '{self.model.oktopk_reduce_method}' not implemented"
                    )
        else:
            self._reduce_topk = self._reduce_topk_local

    def update(self, layer: Layerable, update: bool = True, sync: bool = True) -> None:
        """Optimizer update step for a given layer.

        Args:
            layer: The layer to update.
        """
        if not layer.grad_vars:
            return

        self._select_methods(sync)

        for w_, dw_ in layer.grad_vars.items():
            # Get layer weights and gradients
            w: np.ndarray
            dw: np.ndarray
            w, dw = getattr(layer, w_), getattr(layer, dw_)

            # Compute k from: layer_params * self.density
            k = int(dw.size * self.density)
            k = max(self.min_k_layer or dw.size, k)

            # Initialize current layer-parameter values
            self.local_th = self.all_local_th[layer.id][dw_]
            self.global_th = self.all_global_th[layer.id][dw_]
            self.residuals = self.all_residuals[layer.id][dw_]
            self.velocity = self.all_velocity[layer.id][dw_]
            self.boundaries = self.all_boundaries[layer.id][dw_]

            # Compute acc
            if update:
                acc = self.residuals + dw
            else:
                # TODO: revise if zeros or residuals
                acc = self.residuals.copy()

            # Main part of ok-topk: compute the values
            # that contribute to the update and its indexes
            sparse_u, indexes = self._ok_sparse_allreduce(
                acc, self.iterations[layer.id], k, self.tau, self.tau_prime
            )

            # Update residuals
            residuals = acc.reshape(-1)
            residuals[indexes] = 0
            self.residuals = residuals.reshape(acc.shape)

            # Save for next updates thresholds and boundaries
            self.all_local_th[layer.id][dw_] = self.local_th
            self.all_global_th[layer.id][dw_] = self.global_th
            self.all_residuals[layer.id][dw_] = self.residuals
            self.all_velocity[layer.id][dw_] = self.velocity
            self.all_boundaries[layer.id][dw_] = self.boundaries

            # Perform the weights update
            self._update_weights(w, sparse_u)

        self.iterations[layer.id] += 1

    def _ok_sparse_allreduce(
        self,
        acc: np.ndarray,
        t: int,
        k: int,
        space_repartition_t: int,
        thresholds_re_evaluation_t: int,
    ) -> tuple[SparseFlatArray, np.ndarray]:
        """
        Performs the Ok-Topk sparse allreduce operation.

        This method executes the Ok-Topk sparse allreduce algorithm, which
        optimizes communication by only exchanging the most significant
        gradient values (top-k) across distributed processes. The method
        periodically re-evaluates the thresholds and repartitions the
        gradient space to maintain efficiency and accuracy.

        Parameters:
            acc (np.array): 2D dense gradient matrix accumulation values.
            t (int): Current iteration number.
            k (int): Number of top-k gradient to select in current layer.
            space_repartition_t (int):
                Iterations between space repartitioning.
            thresholds_re_evaluation_t (int):
                Iterations between threshold evaluation.

        Returns:
            out (tuple with two elements:):
                - sparse_u (SparseFlatArray):
                    The updated gradient values in 2D sparse format.
                - indexes (np.array):
                    The indices of the top-k gradient values that were updated.
        """

        if t % thresholds_re_evaluation_t == 0:
            self.local_th = self._th_re_evaluate(acc, k)

        if t % space_repartition_t == 0:
            self.boundaries = self._space_repartition(acc, self.local_th)

        sparse_reduced_region_topk, local_topk_indexes = self._split_and_reduce(
            acc, self.local_th, self.boundaries
        )

        if t % thresholds_re_evaluation_t == 0:
            sparse_all_reduced_topk = self._sparse_allgather(sparse_reduced_region_topk)
            self.global_th = self._th_re_evaluate(sparse_all_reduced_topk.values, k)

        sparse_u, global_topk_indexes = self._balance_and_allgather(
            sparse_reduced_region_topk, self.global_th
        )
        indexes = np.intersect1d(local_topk_indexes, global_topk_indexes, assume_unique=True)
        return sparse_u, indexes

    def _th_re_evaluate(self, array: np.ndarray, k: int) -> float:
        """Re-evaluate k threshold"""
        array = np.abs(array.reshape(-1))

        # fast-path
        if k >= len(array):
            return array.min(initial=0.0)

        return np.partition(array, -k)[-k]

    def _space_repartition(self, acc: np.ndarray, local_th: float) -> BoundaryArray:
        raise ValueError("No partition method selected")

    def _space_repartition_local(self, acc: np.ndarray, local_th: float) -> BoundaryArray:
        self.boundaries.fill(acc.size)

        return self.boundaries

    def _space_repartition_dense(self, acc: np.ndarray, local_th: float) -> BoundaryArray:
        block_size = acc.size // self.model.comm_size

        for i in range(self.model.comm_size):
            self.boundaries[i] = min(block_size * (i + 1), acc.size)

        return self.boundaries

    def _space_repartition_sparse(self, acc: np.ndarray, local_th: float) -> BoundaryArray:
        assert MPI and self.model.comm, "Communicator needed!"

        sparse = SparseFlatArray.from_dense(acc).threshold(local_th)
        block_size = sparse.nnz // self.model.comm_size

        for i in range(self.model.comm_size):
            self.boundaries[i] = sparse.indexes[min(block_size * (i + 1), sparse.nnz - 1)]
        self.boundaries[-1] += 1

        self.model.comm.Allreduce(MPI.IN_PLACE, self.boundaries, op=MPI.SUM)
        self.boundaries /= self.model.comm_size

        return self.boundaries

    def _split_and_reduce(
        self, acc: np.ndarray, local_th: float, boundaries: BoundaryArray
    ) -> tuple[SparseFlatArray, np.ndarray]:
        """
        First main phase of ok_sparse_allreduce.

        Split the gradients into partitions
        and reduce them by selecting top-k values.
        Each worker receives sparse regions
        from the other workers and then
        conducts the reduction locally.

        Parameters:
            acc (np.arrray): 2D gradient matrix accumulation in dense format.
            local_th (float): Local threshold for selecting top-k values.
            boundaries (np.array): Boundaries for partitioning the gradient
                [row_end_p0, row_end_p1, row_end_p2, ...]

        Returns:
            out (tuple with two elements:):
                - sparse_reduced_region_topk (SparseFlatArray):
                    The reduced top-k gradient values in SPARSE format.
                - local_topk_indexes (tuple(np.array, np.array)):
                    The indices of the top-k gradient values selected locally.
        """

        sparse_topk = SparseFlatArray.from_dense(acc).threshold(local_th)
        sparse_reduced_region_topk = self._reduce_topk(sparse_topk, boundaries)
        return sparse_reduced_region_topk, sparse_topk.indexes

    def _reduce_topk(
        self,
        sparse_topk: SparseFlatArray,
        boundaries: BoundaryArray,
    ) -> SparseFlatArray:
        raise ValueError("No reduce method selected")

    def _reduce_topk_local(
        self,
        sparse_topk: SparseFlatArray,
        boundaries: BoundaryArray,
    ) -> SparseFlatArray:
        return sparse_topk

    def _reduce_topk_collective_allreduce_then_slice(
        self, sparse_topk: SparseFlatArray, boundaries: BoundaryArray
    ) -> SparseFlatArray:
        assert MPI and self.model.comm, "Communicator needed!"
        sparse_topk = self.model.comm.allreduce(sparse_topk, op=MPI.SUM)

        start = 0 if self.model.comm_rank == 0 else self.boundaries[self.model.comm_rank - 1]
        end = self.boundaries[self.model.comm_rank]

        sparse_topk = sparse_topk[
            np.searchsorted(sparse_topk.indexes, start): np.searchsorted(sparse_topk.indexes, end)
        ]

        sparse_topk.values *= self.model.rank_weight

        return sparse_topk

    def _reduce_topk_collective_region_wise_reduce_sync(
        self, sparse_topk: SparseFlatArray, boundaries: BoundaryArray
    ) -> SparseFlatArray:
        assert MPI and self.model.comm, "Communicator needed!"
        start = 0

        reduced_regions_sparse: list[SparseFlatArray] = [None] * self.model.comm_size  # pyright: ignore[reportAssignmentType]
        for region in range(self.model.comm_size):
            end = boundaries[region]
            reduced_regions_sparse[region] = self.model.comm.reduce(
                sparse_topk[
                    np.searchsorted(sparse_topk.indexes, start): np.searchsorted(
                        sparse_topk.indexes, end
                    )
                ],
                op=MPI.SUM,
                root=region,
            )
            start = end
        region_reduced_sparse = reduced_regions_sparse[self.model.comm_rank]
        region_reduced_sparse.values *= self.model.rank_weight
        return region_reduced_sparse

    def _reduce_topk_collective_region_wise_reduce_async(
        self, sparse_topk: SparseFlatArray, boundaries: BoundaryArray
    ) -> SparseFlatArray:
        # NOTE: Posible with PyMPI
        raise NotImplementedError(
            "It is not possible with the current mpi4py version to generate a buffer "
            "with indexes and values and operate with them"
        )

    def _reduce_topk_p2p_region_wise_reduce_static_destination(
        self, sparse_topk: SparseFlatArray, boundaries: BoundaryArray
    ) -> SparseFlatArray:
        assert MPI and self.model.comm, "Communicator needed!"
        # Prepare a vector region for storing the partial sums
        region_partial_sum: list[SparseFlatArray] = [None] * self.model.comm_size  # pyright: ignore[reportAssignmentType]
        for region in range(self.model.comm_size):
            start = 0 if region == 0 else boundaries[region - 1]
            end = boundaries[region]
            region_partial_sum[region] = sparse_topk[
                np.searchsorted(sparse_topk.indexes, start): np.searchsorted(
                    sparse_topk.indexes, end
                )
            ]

        # Overlaps comm. steps with computation (sparse sum)
        # On comm_step i: P{rank} sends to P{rank + 1} region{rank - i % nprocs}.
        destination = (self.model.comm_rank + 1) % self.model.comm_size
        receive_from = (self.model.comm_rank - 1) % self.model.comm_size
        for comm_step in range(1, self.model.comm_size):
            region_to_send = (self.model.comm_rank - comm_step) % self.model.comm_size
            region_to_recv = (self.model.comm_rank - comm_step - 1) % self.model.comm_size
            # recv_req = self.model.comm.irecv(source=receive_from)
            # self.model.comm.send(sparse_region_partial_sum[region_to_send], dest=destination)
            # sparse_region_partial_sum[region_to_recv] += recv_req.wait()
            region_partial_sum[region_to_recv] += self.model.comm.sendrecv(
                region_partial_sum[region_to_send], dest=destination, source=receive_from
            )
        region_reduced_sparse = region_partial_sum[self.model.comm_rank]
        region_reduced_sparse.values *= self.model.rank_weight
        return region_reduced_sparse

    def _reduce_topk_p2p_region_wise_reduce_destination_rotation_and_bucketing(
        self, sparse_topk: SparseFlatArray, boundaries: BoundaryArray
    ) -> SparseFlatArray:
        assert MPI and self.model.comm, "Communicator needed!"
        # There are (nprocs - 1) messages to send (excluding self)
        total_sends = self.model.comm_size - 1
        requests: list[Request] = [None] * total_sends

        # Compute local slice of sparse_topk (the "self" region)
        start = 0 if self.model.comm_rank == 0 else boundaries[self.model.comm_rank - 1]
        end = boundaries[self.model.comm_rank]
        sparse_reduced_region = sparse_topk[
            np.searchsorted(sparse_topk.indexes, start): np.searchsorted(sparse_topk.indexes, end)
        ]

        # Process sends and receives in buckets.
        bucket_size = 2
        region = (self.model.comm_rank + 1) % self.model.comm_size
        for comm_step in range(0, total_sends, bucket_size):
            # The current bucket may have fewer messages than bucket_size (i.e. the last bucket)
            current_bucket_size = min(bucket_size, total_sends - comm_step)
            # Non-blocking sends for the current bucket
            for i in range(current_bucket_size):
                start = 0 if region == 0 else boundaries[region - 1]
                end = boundaries[region]
                requests[comm_step + i] = self.model.comm.isend(
                    sparse_topk[
                        np.searchsorted(sparse_topk.indexes, start): np.searchsorted(
                            sparse_topk.indexes, end
                        )
                    ],
                    dest=region,
                )
                region = (region + 1) % self.model.comm_size
            # After sending the bucket, perform the receives sequentially for the same bucket.
            for i in range(current_bucket_size):
                sparse_reduced_region += self.model.comm.recv()

        MPI.Request.Waitall(requests)  # NOTE: not required on PyMPI
        sparse_reduced_region.values *= self.model.rank_weight
        return sparse_reduced_region

    def _sparse_allgather(self, local_data: SparseFlatArray) -> SparseFlatArray:
        """
        Gathers data from all processes.

        NOTE: Concat is fine as indexes come from a disjoint split with global order!

        Parameters:
            local_data (SparseFlatArray): The local data to be gathered.
        Returns:
            gathered_data (SparseFlatArray): The gathered global data in the specified format.
        """
        gathered = self.model.comm.allgather(local_data)
        indexes = np.concat([sparse.indexes for sparse in gathered])
        values = np.concat([sparse.values for sparse in gathered])
        result = SparseFlatArray(local_data.shape, indexes, values)
        assert result.is_canonical(), "gathered arrays not disjoint with global order"
        return result

    def _balance_and_allgather(
        self, sparse_reduced_region_topk: SparseFlatArray, global_th: float
    ) -> tuple[SparseFlatArray, np.ndarray]:
        """
        Second main phase of ok_sparse_allreduce.

        Performs the allgather of the sparse_reduced_region_topk
        values among workers.

        Parameters:
            sparse_reduced_region_topk (SparseFlatArray):
                a 2D sparse gradient matrix.
            global_th (float):
                the global threshold to perfrom top selection.

        Returns:
            out (tuple with two elements):
                - sparse_allgather_topk (SparseFlatArray):
                    A 2D sparse gradient matrix with
                    the global top-k selection.
                - reduced_region_global_topk_indexes
                  (tuple(np.array, np.array)):
                    The indices of the top-k gradient
                    values region reduced.
        """

        # 1. Global topk selection
        sparse_reduced_region_global_topk = sparse_reduced_region_topk.threshold(global_th)
        assert sparse_reduced_region_global_topk

        # 2. Data packaging
        # TODO

        # 3. Data balancing
        # TODO

        # 4. Allgatherv using recursive doubling
        sparse_allgather_topk = self._sparse_allgather(sparse_reduced_region_global_topk)
        return sparse_allgather_topk, sparse_allgather_topk.indexes

    def _update_weights(
        self,
        w: np.ndarray,
        sparse_u: SparseFlatArray,
    ) -> None:
        """
        Update weights and set to weight layer attribute.

        w -= (u / self.model.comm_size)
        setattr(layer, w_type, w)

        Parameters:
            layer (int): layer id
            w_type (string): weight param type (bias, weight, ...)
            w (np.array): N dimensional dense weights matrix/tensor
            sparse_u (SparseFlatArray): Sparse 2D gradient matrix in
                SPARSE format to update w

        Returns:
            (void): instead it directly applies the result
                to the weight layer attribute
        """

        velocity = self.velocity
        velocity *= self.momentum
        velocity = velocity.reshape(-1)
        velocity[sparse_u.indexes] += sparse_u.values
        velocity = velocity.reshape(w.shape)
        w -= self.learning_rate * (self.decay * w + velocity)
        self.velocity = velocity  # NOTE: GPU (reshape may copy)
