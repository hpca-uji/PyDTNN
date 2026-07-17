"""Module for the OkTopkSP optimizer implementation using NumPy."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from pydtnn.abstract.layerable import Layerable
from pydtnn.backends.numpy.optimizers.abstract.optimizer import OptimizerNumpy
from pydtnn.libs import numpy as np
from pydtnn.optimizers.oktopksp import OkTopkSP
from pydtnn.utils.sparse.sparse import SparseMatrixFlat

__all__ = ("OkTopkSPNumpy",)

logger = logging.getLogger(__name__)


if TYPE_CHECKING:
    import numpy as np  # noqa: F811 (override typing)
    from pympi.MPI import Request  # type: ignore


try:
    from pydtnn.libs.mpi import MPI
except (ImportError, ModuleNotFoundError):
    pass

type AllGatherTypes = (
    np.ndarray[tuple[int, ...], np.dtype[np.float32 | np.float64]]
    | SparseMatrixFlat
)


class OkTopkSPNumpy(OkTopkSP[np.ndarray], OptimizerNumpy):
    """NumPy-based implementation of the OkTopkSP optimizer."""

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
        self.all_boundaries: dict[int, dict[str, np.ndarray]] = {}

        for layer in layers:
            self.iterations[layer.id] = 0

            # The following attributes will be initialized later.
            self.all_local_th[layer.id] = {dw_: None for dw_ in layer.grad_vars.values()}  # type: ignore
            self.all_global_th[layer.id] = {dw_: None for dw_ in layer.grad_vars.values()}  # type: ignore
            self.all_residuals[layer.id] = {dw_: None for dw_ in layer.grad_vars.values()}  # type: ignore
            self.all_boundaries[layer.id] = {dw_: None for dw_ in layer.grad_vars.values()}  # type: ignore

    def update(self, layer: Layerable) -> None:
        """Optimizer update step for a given layer.

        Args:
            layer: The layer to update.
        """
        for w_, dw_ in layer.grad_vars.items():
            # Get layer weights and gradients
            w, dw = getattr(layer, w_), getattr(layer, dw_)
            w: np.ndarray
            dw: np.ndarray

            # Reshape dw to 2D matrix
            self.dw_original_shape = dw.shape
            if len(self.dw_original_shape) != 2:
                dw = dw.reshape(dw.shape[0], -1)
            self.dw_2d_shape = dw.shape

            # Compute k from: layer_params * self.density
            k = int(np.prod(self.dw_original_shape) * self.density)
            k = self.min_k_layer if k < self.min_k_layer else k

            # Initialize current layer-parameter values
            self.local_th = self.all_local_th[layer.id][dw_]
            self.global_th = self.all_global_th[layer.id][dw_]
            self.boundaries = self.all_boundaries[layer.id][dw_]
            if self.all_residuals[layer.id][dw_] is None:
                self.all_residuals[layer.id][dw_] = np.zeros_like(dw, dtype=layer.model.dtype)

            # Compute acc
            acc = self.all_residuals[layer.id][dw_] + dw

            # Main part of ok-topk: compute the values
            # that contribute to the update and its indexes
            coo_u, indexes = self._ok_sparse_allreduce(
                acc, self.iterations[layer.id], k, self.tau, self.tau_prime
            )

            # Update residuals
            residuals = acc
            residuals[indexes] = 0
            self.all_residuals[layer.id][dw_] = residuals

            # Save for next updates thresholds and boundaries
            self.all_local_th[layer.id][dw_] = self.local_th
            self.all_global_th[layer.id][dw_] = self.global_th
            self.all_boundaries[layer.id][dw_] = self.boundaries

            # Perform the weights update
            self._update_weights(layer, w_, w, coo_u)

        self.iterations[layer.id] += 1

    def _update_weights(
        self,
        layer: Layerable,
        w_type: str,
        w: np.ndarray,
        coo_u: SparseMatrixFlat,
    ) -> None:
        """
        Update weights and set to weight layer attribute.

        w -= (u / self.model.nprocs)
        setattr(layer, w_type, w)

        Parameters:
            layer (int): layer id
            w_type (string): weight param type (bias, weight, ...)
            w (np.array): N dimensional dense weights matrix/tensor
            coo_u (SparseMatrixCOO): Sparse 2D gradient matrix in
                COO format to update w

        Returns:
            (void): instead it directly applies the result
                to the weight layer attribute
        """

        if len(self.dw_original_shape) != 2:
            w = w.reshape(w.shape[0], -1)
        coo_u.data /= self.model.nprocs
        velocity = getattr(layer, "velocity_%s" % w_type, np.zeros_like(w, dtype=layer.model.dtype))
        velocity *= self.momentum
        velocity[coo_u.row, coo_u.col] += coo_u.data
        w -= self.learning_rate * (self.decay * w + velocity)
        if len(self.dw_original_shape) != 2:
            w = w.reshape(self.dw_original_shape)
        setattr(layer, w_type, w)
        setattr(layer, "velocity_%s" % w_type, velocity)

    def _ok_sparse_allreduce(
        self,
        acc: np.ndarray,
        t: int,
        k: int,
        space_repartition_t: int,
        thresholds_re_evaluation_t: int,
    ) -> tuple[SparseMatrixFlat, np.ndarray]:
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
                - coo_u (SparseMatrixCOO):
                    The updated gradient values in 2D sparse format.
                - indexes (np.array):
                    The indices of the top-k gradient values that were updated.
        """

        if t % thresholds_re_evaluation_t == 0:
            self.local_th = self._th_re_evaluate_dense(acc, k)

        if t % space_repartition_t == 0:
            self.boundaries = self._space_repartition(acc, self.local_th)

        coo_reduced_region_topk, local_topk_indexes = self._split_and_reduce(
            acc, self.local_th, self.boundaries
        )

        if t % thresholds_re_evaluation_t == 0:
            coo_all_reduced_topk = self._allgather_sparse_matrix(coo_reduced_region_topk)
            self.global_th = self._th_re_evaluate_coo(coo_all_reduced_topk, k)

        coo_u, global_topk_indexes = self._balance_and_allgather(
            coo_reduced_region_topk, self.global_th
        )
        indexes = SparseMatrixFlat.intersection_indexes(local_topk_indexes, global_topk_indexes)
        return coo_u, indexes

    def _th_re_evaluate_partition(self, flat_matrix: np.ndarray, k: int) -> float:
        if k > len(flat_matrix):
            return flat_matrix.min()
        threshold = np.partition(flat_matrix, -k)[-k]
        return threshold

    def _th_re_evaluate_dense(
        self,
        matrix: np.ndarray,
        k: int,
    ) -> float:
        """
        Return the absolute gradient threshold for a given matrix.

        Parameters:
            matrix (np.array or SparseMatrixCOO):
                A 2D gradient matrix, in np.array for 'dense' input_format
                or SparseMatrixCOO for 'coo' input_format.
            k (int): Indicating the number of top gradient values to consider.

        Returns:
            threshold (float):
                The absolute gradient threshold based on the top k values.
        """

        if k <= 0:
            return 0.0

        return self._th_re_evaluate_partition(np.abs(matrix).flatten(), k)

    def _th_re_evaluate_coo(
        self,
        matrix: SparseMatrixFlat,
        k: int,
    ) -> float:
        """
        Return the absolute gradient threshold for a given matrix.

        Parameters:
            matrix (np.array or SparseMatrixCOO): A 2D gradient matrix, in np.array for 'dense' input_format or
                SparseMatrixCOO for 'coo' input_format.
            k (int): Indicating the number of top gradient values to consider.

        Returns:
            threshold (float): The absolute gradient threshold based on the top k values.
        """

        if k <= 0:
            return 0.0

        if matrix.number_non_zeros == 0:
            return 1.0

        return self._th_re_evaluate_partition(np.abs(matrix.data), k)

    def _space_repartition(
        self, acc: np.ndarray, local_th: float, balanced: bool = True
    ) -> np.ndarray:
        """
        Returns the boundaries of the regions of the
        gradient matrix for the split and reduce phase.

        Parameters:
            acc (np.array): 2D dense gradient matrix values
            local_th (float): local process gradient threshold
            balanced (boolean, optional):
                If not balanced a static row partition is performed,
                if balanced a topk gradiend distribution is considered
                in the row partition.

        Warning:
            Balanced space repartition does not provide
            the same exact accuracy as static space repartition.

        Returns:
            boundaries (np.array): [row_end_p0, row_end_p1, row_end_p2, ...]
        """

        logger.debug(
            f"In '_space_repartition', balanced = '{balanced}' is being used"
        )

        output = None

        if not balanced:
            boundaries = np.zeros(self.model.nprocs, dtype=np.int32)
            total_rows = self.dw_original_shape[0]
            block_size = total_rows // self.model.nprocs
            for i in range(0, self.model.nprocs - 1):
                boundaries[i] = block_size * (i + 1)
            boundaries[self.model.nprocs - 1] = total_rows

            output = boundaries
        else:
            coo_topk = SparseMatrixFlat.from_dense_top_selection(acc, local_th)

            current_row = 0
            current_proc = 0
            rows = coo_topk.row
            topk_in_current_proc = 0
            total_rows = coo_topk.shape[0]
            boundaries = np.zeros(self.model.nprocs, dtype=np.int32)
            topk_per_proc = coo_topk.number_non_zeros // self.model.nprocs
            topk_per_row = np.zeros(total_rows, dtype=np.int32)
            np.add.at(topk_per_row, rows, 1)  # type: ignore

            while current_proc < self.model.nprocs - 1:
                if current_row < total_rows:
                    topk_in_current_proc += topk_per_row[current_row]
                    if topk_in_current_proc >= topk_per_proc:
                        boundaries[current_proc] = current_row
                        topk_in_current_proc = 0
                        current_proc += 1
                    current_row += 1
                else:
                    boundaries[current_proc] = current_row
                    current_proc += 1
            boundaries[self.model.nprocs - 1] = total_rows

            global_boundaries = (
                self.model.comm.allreduce(boundaries, op=MPI.SUM) // self.model.nprocs
            )
            output = global_boundaries

        return output

    def _split_and_reduce(
        self, acc: np.ndarray, local_th: float, boundaries: np.ndarray
    ) -> tuple[SparseMatrixFlat, np.ndarray]:
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
                - coo_reduced_region_topk (SparseMatrixCOO):
                    The reduced top-k gradient values in COO format.
                - local_topk_indexes (tuple(np.array, np.array)):
                    The indices of the top-k gradient values selected locally.
        """

        coo_topk = SparseMatrixFlat.from_dense_top_selection(acc, local_th)
        coo_reduced_region_topk = self._reduce_topk(coo_topk, boundaries)
        return coo_reduced_region_topk, coo_topk.indexes

    def _balance_and_allgather(
        self, coo_reduced_region_topk: SparseMatrixFlat, global_th: float
    ) -> tuple[SparseMatrixFlat, np.ndarray]:
        """
        Second main phase of ok_sparse_allreduce.

        Performs the allgather of the coo_reduced_region_topk
        values among workers.

        Parameters:
            coo_reduced_region_topk (SparseMatrixCOO):
                a 2D sparse gradient matrix.
            global_th (float):
                the global threshold to perfrom top selection.

        Returns:
            out (tuple with two elements):
                - coo_allgather_topk (SparseMatrixCOO):
                    A 2D sparse gradient matrix with
                    the global top-k selection.
                - reduced_region_global_topk_indexes
                  (tuple(np.array, np.array)):
                    The indices of the top-k gradient
                    values region reduced.
        """

        # 1. Global topk selection
        coo_reduced_region_global_topk = coo_reduced_region_topk.threshold_selection(
            global_th, inplace=False
        )
        assert coo_reduced_region_global_topk

        # 2. Data packaging
        # TODO

        # 3. Data balancing
        # TODO

        # 4. Allgatherv using recursive doubling
        coo_allgather_topk = self._allgather_sparse_matrix(coo_reduced_region_global_topk)
        return coo_allgather_topk, coo_reduced_region_global_topk.indexes

    def _intersect_indexes(
        self,
        local_indexes: np.ndarray,
        global_indexes: np.ndarray,
    ) -> np.ndarray:
        """
        Calculates the intersection of two sets of indices of 2D.

        The assertion statement is only executed when the script
        is not run in optimized mode (python3 -O script.py).
        Remember that '_has_canonical_format' should only be
        used for debugging/development purposes
        to assert that indexes are correct.

        Indexes in scipy are usually in canonical format,
        so it should not be necessary to evaluate the indexes format.
        When optimized mode is enabled (python3 -O script.py),
        the assert sentences are not computed.

        Parameters:
            local_indexes (np.array): an array representing the indices,
            global_indexes (np.array): an array representing the indices,

        Returns:
            intersected_indexes (np.array): a np.array of common indices.

        Example:
            - local_indexes  = np.array([0, 1, 2, 3, 5, 8])
            - global_indexes = np.array([1, 3, 8, 13, 21]
            - output: array([1, 3, 8])
        """

        return np.intersect1d(local_indexes, global_indexes, assume_unique=True)

    def _reduce_topk_collective_allreduce_then_slice(
        self, coo_topk: SparseMatrixFlat, boundaries: np.ndarray
    ) -> SparseMatrixFlat:
        logger.warning(
            "reduce_topk_collective_allreduce_then_slice' should be used only "
            "in case of debugging for performance reasons."
        )

        assert self.model.comm, "Communicator needed!"
        all_reduced_coo = self.model.comm.allreduce(coo_topk, op=MPI.SUM)
        row_start = 0 if self.model.rank == 0 else boundaries[self.model.rank - 1]
        row_end = boundaries[self.model.rank]
        region_reduced_coo = all_reduced_coo.slice_selection(row_start, row_end)
        region_reduced_coo.data /= self.model.nprocs
        return region_reduced_coo

    def _reduce_topk_collective_region_wise_reduce_sync(
        self, coo_topk: SparseMatrixFlat, boundaries: np.ndarray
    ) -> SparseMatrixFlat:
        assert self.model.comm, "Communicator needed!"
        row_start = 0
        # # type: ignore (The values will be set later)
        reduced_regions_coo: list[SparseMatrixFlat] = [None] * self.model.nprocs  # type: ignore
        for region in range(self.model.nprocs):
            row_end = boundaries[region]
            reduced_regions_coo[region] = self.model.comm.reduce(
                coo_topk.slice_selection(row_start, row_end), op=MPI.SUM, root=region
            )
            row_start = row_end
        region_reduced_coo = reduced_regions_coo[self.model.rank]
        region_reduced_coo.data /= self.model.nprocs
        return region_reduced_coo

    def _reduce_topk_collective_region_wise_reduce_async(
        self, coo_topk: SparseMatrixFlat, boundaries: np.ndarray
    ) -> SparseMatrixFlat:
        raise NotImplementedError(
            "It is not possible with the current mpi4py version to generate a buffer "
            "with indexes and values and operate with them"
        )

    def _reduce_topk_p2p_region_wise_reduce_static_destination(
        self, coo_topk: SparseMatrixFlat, boundaries: np.ndarray
    ) -> SparseMatrixFlat:
        assert self.model.comm, "Communicator needed!"
        # Prepare a vector region for storing the partial sums
        coo_region_partial_sum: list[SparseMatrixFlat] = [None] * self.model.nprocs  # type: ignore
        for region in range(self.model.nprocs):
            row_start = 0 if region == 0 else boundaries[region - 1]
            row_end = boundaries[region]
            coo_region_partial_sum[region] = coo_topk.slice_selection(row_start, row_end)

        # Overlaps comm. steps with computation (sparse sum)
        # On comm_step i: P{rank} sends to P{rank + 1} region{rank - i % nprocs}.
        destination = (self.model.rank + 1) % self.model.nprocs
        receive_from = (self.model.rank - 1) % self.model.nprocs
        for comm_step in range(1, self.model.nprocs):
            region_to_send = (self.model.rank - comm_step) % self.model.nprocs
            region_to_recv = (self.model.rank - comm_step - 1) % self.model.nprocs
            # recv_req = self.model.comm.irecv(source=receive_from)
            # self.model.comm.send(coo_region_partial_sum[region_to_send], dest=destination)
            # coo_region_partial_sum[region_to_recv] += recv_req.wait()
            coo_region_partial_sum[region_to_recv] += self.model.comm.sendrecv(
                coo_region_partial_sum[region_to_send], dest=destination, source=receive_from
            )
        region_reduced_coo = coo_region_partial_sum[self.model.rank]
        region_reduced_coo.data /= self.model.nprocs

    def _reduce_topk_p2p_region_wise_reduce_destination_rotation_and_bucketing(
        self, coo_topk: SparseMatrixFlat, boundaries: np.ndarray
    ) -> SparseMatrixFlat:

        assert self.model.comm, "Communicator needed!"
        # There are (nprocs - 1) messages to send (excluding self)
        total_sends = self.model.nprocs - 1
        requests: list[Request] = [None] * total_sends  # type: ignore

        # Compute local slice of coo_topk (the "self" region)
        row_start = 0 if self.model.rank == 0 else boundaries[self.model.rank - 1]
        row_end = boundaries[self.model.rank]
        coo_reduced_region = coo_topk.slice_selection(row_start, row_end)

        # Process sends and receives in buckets.
        bucket_size = 2
        region = (self.model.rank + 1) % self.model.nprocs
        for comm_step in range(0, total_sends, bucket_size):
            # The current bucket may have fewer messages than bucket_size (i.e. the last bucket)
            current_bucket_size = min(bucket_size, total_sends - comm_step)
            # Non-blocking sends for the current bucket
            for i in range(current_bucket_size):
                row_start = 0 if region == 0 else boundaries[region - 1]
                row_end = boundaries[region]
                requests[comm_step + i] = self.model.comm.isend(
                    coo_topk.slice_selection(row_start, row_end), dest=region
                )
                region = (region + 1) % self.model.nprocs
            # After sending the bucket, perform the receives sequentially for the same bucket.
            for i in range(current_bucket_size):
                coo_reduced_region += self.model.comm.recv()

        MPI.Request.Waitall(requests)
        coo_reduced_region.data /= self.model.nprocs
        return coo_reduced_region

    # TODO: Move this to different methods.
    def _reduce_topk(
        self,
        coo_topk: SparseMatrixFlat,
        boundaries: np.ndarray,
        method: str = "p2p_region_wise_reduce_destination_rotation_and_bucketing",
    ) -> SparseMatrixFlat:
        """
        Reduce the topk elements in regions defined by boundaries.

        Parameters:
            coo_topk (SparseMatrixCOO): a 2D sparse array in COO format with the values and indexes of topk.
             boundaries (np.array): boundaries for partitioning the gradient space like
             [row_end_p0, row_end_p1, row_end_p2, ...].
            method (str, optional): The method to use for reduce topk

        Returns:
            coo_reduced_region (SparseMatrixCOO): The reduced topk values in COO format.
        """
        logger.debug(
            f"In '_reduce_topk', the method that it is being used is '{method}'"
        )

        if self.model.nprocs == 1:
            return coo_topk

        assert self.model.comm, "Communicator need!"

        match method:
            case "collective_allreduce_then_slice":
                return self._reduce_topk_collective_allreduce_then_slice(
                    coo_topk, boundaries
                )
            case "collective_region_wise_reduce_sync":
                return self._reduce_topk_collective_region_wise_reduce_sync(
                    coo_topk, boundaries
                )
            case "collective_region_wise_reduce_async":
                return self._reduce_topk_collective_region_wise_reduce_async(
                    coo_topk, boundaries
                )
            case "p2p_region_wise_reduce_static_destination":
                return self._reduce_topk_p2p_region_wise_reduce_static_destination(
                    coo_topk, boundaries
                )
            case "p2p_region_wise_reduce_destination_rotation_and_bucketing":
                return self._reduce_topk_p2p_region_wise_reduce_destination_rotation_and_bucketing(
                    coo_topk, boundaries
                )
            case _:
                raise NotImplementedError(f"Method '{method}' not implemented")

    def _allgather_sparse_matrix(self, local_data: SparseMatrixFlat) -> SparseMatrixFlat:
        """
        Gathers data from all processes.

        NOTE: Concat is fine as indexes come from a disjoint split with global order!

        Parameters:
            local_data (SparseMatrixCOO): The local data to be gathered.
        Returns:
            gathered_data (SparseMatrixCOO): The gathered global data in the specified format.
        """
        gathered = self.model.comm.allgather(local_data.get_data_and_indexes())
        all_val = np.concatenate([t[0] for t in gathered])
        all_indexes = np.concatenate([t[1] for t in gathered])
        return SparseMatrixFlat(all_val, all_indexes, self.dw_2d_shape)

    def _has_canonical_format(self, indexes: np.ndarray) -> bool:
        """
        Check if indexes follows the COO canonical format.

        Format:
            - Indexes are sorted by row and then by column
            - There are no duplicate entries

        This function is computationally expensive and therefore should only be used for developing/debugging purposes.
        This function should only be used in developement to assert that sparse matrices have canonical format.

        Parameters:
            indexes (np.ndarray): indexes to check

        Returns:
            has_canonical_format (bool): True if indexes are in canonical format, False if not.
        """

        logger.warning(
            "This function ('has_canonical_format') should be used only in case of debugging for"
            " performance reasons."
        )

        return (len(indexes) == 0) or bool(np.all(indexes[:-1] < indexes[1:]))
