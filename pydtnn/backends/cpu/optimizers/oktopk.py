import warnings

import numpy as np

from pydtnn.backends.cpu.optimizers.optimizer import OptimizerCPU
from pydtnn.backends.cpu.utils.oktopk_utils_cython import compute_dense_acc_cython, intersect_2d_indexes_cython, reset_residuals_cython, update_sparsed_weights_cython, update_sparsed_weights_mv_cython
from pydtnn.layer_base import LayerBase
from pydtnn.optimizers.oktopk import OkTopk
from pydtnn.utils.sparse import SparseMatrixCOO

try:
    from pydtnn.libs.libmpi import MPI
except (ImportError, ModuleNotFoundError):
    pass


class OkTopkCPU(OkTopk[np.ndarray], OptimizerCPU):
    def initialize(self, list_layers: list[LayerBase]) -> None:
        super().initialize(list_layers)

        for layer in list_layers:
            self.iterations[layer.id] = 0
            self.all_local_th[layer.id] = {dw_: None for dw_ in layer.grad_vars.values()}
            self.all_global_th[layer.id] = {dw_: None for dw_ in layer.grad_vars.values()}
            self.all_residuals[layer.id] = {dw_: None for dw_ in layer.grad_vars.values()}
            self.all_boundaries[layer.id] = {dw_: None for dw_ in layer.grad_vars.values()}

    def update(self, layer):
        for w_, dw_ in layer.grad_vars.items():
            # Get layer weights and gradients
            w, dw = getattr(layer, w_), getattr(layer, dw_)

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
            acc = self._compute_acc(self.all_residuals[layer.id][dw_], dw, self.learning_rate)

            # Main part of ok-topk: compute the values that contribute to the update and its indexes
            coo_u, indexes = self._ok_sparse_allreduce(acc, self.iterations[layer.id], k, self.tau, self.tau_prime)

            # Update residuals
            self.all_residuals[layer.id][dw_] = self._reset_residuals(acc, indexes)

            # Save for next updates thresholds and boundaries
            self.all_local_th[layer.id][dw_] = self.local_th
            self.all_global_th[layer.id][dw_] = self.global_th
            self.all_boundaries[layer.id][dw_] = self.boundaries

            # Perform the weights update
            self._update_weights(layer, w_, w, coo_u)

        self.iterations[layer.id] += 1

    def _compute_acc(self, residuals, dw, learning_rate, method="cython"):
        """
        Compute acc, where: acc = residuals + (learning_rate * dw)

        Parameters:
            residuals (np.array): 2D dense matrix with the current layer residuals
            dw (np.array): 2D dense matrix with the current layer gradients
            learning_rate (float): learning rate float value
            method (string, optional): The method to use for updating the weights. It can be 'cython' or 'numpy'. Default is 'cython'.

        Warning:
            'cython' method does not provide the same exact accuracy as 'numpy'.

        Returns:
            acc (np.array): 2D dense matrix with the updated residuals
        """

        self._show_message_only_once(f"\n\nIn '_compute_acc', the method that it is being used is '{method}'")

        if method == "cython":
            return compute_dense_acc_cython(residuals, dw, learning_rate)

        if method == "numpy":
            return residuals + (learning_rate * dw)

        raise NotImplementedError(f"Method '{method}' not implemented")

    def _reset_residuals(self, acc, indexes, method="cython"):
        """
        Update residuals: set zero value if it is in indexes, else acc value is set.
        If density is 100% and some gradients are zero, scipy will be removing those indexes even if no sparsity is applied.
        Thus, to simulate 100% density, residuals must be always zero.
        This means that a slightly sparse factor will may remove more values because the gradients are already zero.

        Parameters:
            acc (np.array): 2D dense matrix
            indexes (tuple(np.array, np.array)): a tuple with rows and cols
            method (string, optional): The method to use for updating the weights. It can be 'cython' or 'numpy'. Default is 'cython'.

        Returns:
            residuals (np.array): which is the same as acc with the values in indexes set to zero.
        """

        self._show_message_only_once(f"In '_reset_residuals', the method that it is being used is '{method}'")

        if self.density == 1:
            return np.zeros_like(acc)

        if method == "cython":
            assert (self._has_canonical_format(indexes))
            return reset_residuals_cython(acc, indexes[0], indexes[1])

        if method == "numpy":
            if len(indexes[0]) > 0:
                acc[indexes] = 0
            return acc

        raise NotImplementedError(f"Method '{method}' not implemented")

    def _update_weights(self, layer, w_type, w, coo_u, method="cython"):
        """
        Update weights: w -= (u / self.model.nprocs)
        and set to weight layer attribute: setattr(layer, w_type, w)

        Parameters:
            layer (int): layer id
            w_type (string): weight param type (bias, weight, ...)
            w (np.array): N dimensional dense weights matrix/tensor
            coo_u (SparseMatrixCOO): Sparse 2D gradient matrix in COO format to update w
            method (string, optional): The method to use for updating the weights. It can be 'cython' or 'numpy'. Default is 'cython'.

        Returns:
            (void): instead it directly applies the result to the weight layer attribute
        """

        self._show_message_only_once(f"In '_update_weights', the method that it is being used is '{method}'")

        if method == "cython":
            if len(self.dw_original_shape) != 2:
                w = w.reshape(w.shape[0], -1)
            w = update_sparsed_weights_cython(w, coo_u.data, coo_u.row, coo_u.col)
            if len(self.dw_original_shape) != 2:
                w = w.reshape(self.dw_original_shape)
            setattr(layer, w_type, w)
            return

        if method == "cython_with_vel_and_momentum":
            if self.momentum == 0:
                warnings.warn("If momentum is 0 use 'cython' method, it produces the same output but it is faster", RuntimeWarning)

            if len(self.dw_original_shape) != 2:
                w = w.reshape(w.shape[0], -1)
            velocity = getattr(layer, "velocity_%s" % w_type, np.zeros_like(w, dtype=layer.model.dtype))
            w, velocity = update_sparsed_weights_mv_cython(w, coo_u.data, coo_u.row, coo_u.col, velocity, self.momentum)
            if len(self.dw_original_shape) != 2:
                w = w.reshape(self.dw_original_shape)
            setattr(layer, w_type, w)
            setattr(layer, "velocity_%s" % w_type, velocity)
            return

        if method == "numpy":
            if len(self.dw_original_shape) != 2:
                w = w.reshape(w.shape[0], -1)
            w[coo_u.row, coo_u.col] -= coo_u.data
            if len(self.dw_original_shape) != 2:
                w = w.reshape(self.dw_original_shape)
            setattr(layer, w_type, w)
            return

        if method == "numpy_with_vel_and_momentum":
            if self.momentum == 0:
                warnings.warn("If momentum is 0 use just 'numpy' method, it produces the same output but it is faster", RuntimeWarning)

            if len(self.dw_original_shape) != 2:
                w = w.reshape(w.shape[0], -1)
            velocity = getattr(layer, "velocity_%s" % w_type, np.zeros_like(w, dtype=layer.model.dtype))
            velocity *= self.momentum
            velocity[coo_u.row, coo_u.col] += coo_u.data
            w[coo_u.row, coo_u.col] -= velocity[coo_u.row, coo_u.col]
            if len(self.dw_original_shape) != 2:
                w = w.reshape(self.dw_original_shape)
            setattr(layer, w_type, w)
            setattr(layer, "velocity_%s" % w_type, velocity)
            return

        if method == "like_sgd":
            """Use only for debugging purposes"""
            warnings.warn("This method should be used only in case of debugging for performance reasons.", RuntimeWarning)

            dw = coo_u.toarray()
            if len(self.dw_original_shape) != 2:
                dw = dw.reshape(self.dw_original_shape)
            velocity = getattr(layer, "velocity_%s" % w_type, np.zeros_like(w, dtype=layer.model.dtype))
            velocity = self.momentum * velocity + dw
            w -= velocity  # Oktopk already computes acc with learning_rate
            setattr(layer, w_type, w)
            setattr(layer, "velocity_%s" % w_type, velocity)
            return

        raise NotImplementedError(f"Method '{method}' not implemented")

    def _ok_sparse_allreduce(self, acc, t, k, space_repartition_t, thresholds_re_evaluation_t):
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
            k (int): Number of top-k gradient values to select in the current layer.
            space_repartition_t (int): Interval of iterations for space repartitioning.
            thresholds_re_evaluation_t (int): Interval of iterations for threshold re-evaluation.

        Returns:
            out (tuple with two elements:):
                - coo_u (SparseMatrixCOO): The updated gradient values in 2D sparse format.
                - indexes (tuple(np.array, np.array)): The indices of the top-k gradient values that were updated.
        """

        if t % thresholds_re_evaluation_t == 0:
            self.local_th = self._th_re_evaluate(acc, k, input_format="dense")

        if t % space_repartition_t == 0:
            self.boundaries = self._space_repartition(acc, self.local_th)

        coo_reduced_region_topk, local_topk_indexes = self._split_and_reduce(acc, self.local_th, self.boundaries)

        if t % thresholds_re_evaluation_t == 0:
            coo_all_reduced_topk = self._allgather(coo_reduced_region_topk)
            self.global_th = self._th_re_evaluate(coo_all_reduced_topk, k, input_format="coo")

        coo_u, global_topk_indexes = self._balance_and_allgather(coo_reduced_region_topk, self.global_th)
        indexes = self._intersect_indexes(local_topk_indexes, global_topk_indexes)
        return coo_u, indexes

    def _th_re_evaluate(self, matrix, k, input_format=None, method="numpy_sort"):
        """
        Return the absolute gradient threshold for a given matrix.

        Parameters:
            matrix (np.array or SparseMatrixCOO): A 2D gradient matrix, in np.array for 'dense' input_format or SparseMatrixCOO for 'coo' input_format.
            k (int): Indicating the number of top gradient values to consider.
            input_format (string): Either 'dense' for a dense matrix or 'coo' for a sparse matrix in COO format.
            method (string, optional): The method to use for threshold selection. It can be 'numpy_sort' or 'numpy_partition'.

        Returns:
            threshold (float): The absolute gradient threshold based on the top k values.
        """

        self._show_message_only_once(f"In '_th_re_evaluate', the method that it is being used is '{method}'")

        if k <= 0:
            return 0.0

        if input_format == "coo" and matrix.nnz == 0:
            return 1.0

        if input_format == "dense" and method == "numpy_sort":
            sorted_matrix = np.sort(np.abs(matrix).flatten())
            threshold = sorted_matrix[max(-k, -len(sorted_matrix))]
            return threshold

        if input_format == "coo" and method == "numpy_sort":
            sorted_data = np.sort(np.abs(matrix.data))
            threshold = sorted_data[max(-k, -len(sorted_data))]
            return threshold

        if input_format == "dense" and method == "numpy_partition":
            flat_matrix = np.abs(matrix).flatten()
            if k > len(flat_matrix):
                return flat_matrix.min()
            threshold = np.partition(flat_matrix, -k)[-k]
            return threshold

        if input_format == "coo" and method == "numpy_partition":
            flat_matrix = np.abs(matrix.data)
            if k > len(flat_matrix):
                return flat_matrix.min()
            threshold = np.partition(flat_matrix, -k)[-k]
            return threshold

        raise NotImplementedError(f"Method '{method}' with format '{input_format}' not implemented")

    def _space_repartition(self, acc, local_th, balanced=True):
        """
        Returns the boundaries of the regions of the gradient matrix for the split and reduce phase.

        Parameters:
            acc (np.array): 2D dense gradient matrix values
            local_th (float): local process gradient threshold
            balanced (boolean, optional): if not balanced a static row partition is performed,
                                          if balanced a topk gradiend distribution is considered in the row partition

        Warning:
            Balanced space repartition does not provide the same exact accuracy as static space repartition.

        Returns:
            boundaries (np.array): [row_end_p0, row_end_p1, row_end_p2, ...]
        """

        self._show_message_only_once(f"In '_space_repartition', balanced = '{balanced}' is being used")

        if not balanced:
            boundaries = np.zeros(self.model.nprocs, dtype=np.int32)
            total_rows = self.dw_original_shape[0]
            block_size = total_rows // self.model.nprocs
            for i in range(0, self.model.nprocs - 1):
                boundaries[i] = block_size * (i + 1)
            boundaries[self.model.nprocs - 1] = total_rows
            return boundaries

        if balanced:
            coo_topk = SparseMatrixCOO.from_dense_top_selection(acc, local_th)

            current_row = 0
            current_proc = 0
            rows = coo_topk.row
            topk_in_current_proc = 0
            total_rows = coo_topk.shape[0]
            boundaries = np.zeros(self.model.nprocs, dtype=np.int32)
            topk_per_proc = coo_topk.nnz // self.model.nprocs
            topk_per_row = np.zeros(total_rows, dtype=np.int32)
            np.add.at(topk_per_row, rows, 1)

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

            global_boundaries = self.model.comm.allreduce(boundaries, op=MPI.SUM) // self.model.nprocs

            return global_boundaries

    def _split_and_reduce(self, acc, local_th, boundaries):
        """
        First main phase of ok_sparse_allreduce.
        Split the gradients into partitions and reduce them by selecting top-k values.
        Each worker receives sparse regions from the other workers and and then conducts the reduction locally.

        Parameters:
            acc (np.arrray): 2D gradient matrix accumulation values in dense format.
            local_th (float): Local threshold for selecting top-k values.
            boundaries (np.array): Boundaries for partitioning the gradient space like [row_end_p0, row_end_p1, row_end_p2, ...]

        Returns:
            out (tuple with two elements:):
                - coo_reduced_region_topk (SparseMatrixCOO): The reduced top-k gradient values in COO format.
                - local_topk_indexes (tuple(np.array, np.array)): The indices of the top-k gradient values selected locally.
        """

        coo_topk = SparseMatrixCOO.from_dense_top_selection(acc, local_th)
        coo_reduced_region_topk = self._reduce_topk(coo_topk, boundaries)
        return coo_reduced_region_topk, coo_topk.get_indexes()

    def _balance_and_allgather(self, coo_reduced_region_topk, global_th):
        """
        Second main phase of ok_sparse_allreduce.
        Performs the allgather of the coo_reduced_region_topk values among workers.

        Parameters:
            coo_reduced_region_topk (SparseMatrixCOO): a 2D sparse gradient matrix.
            global_th (float): the global threshold to perfrom top selection.

        Returns:
            out (tuple with two elements:):
                - coo_allgather_topk (SparseMatrixCOO): A 2D sparse gradient matrix with the global top-k selection.
                - reduced_region_global_topk_indexes (tuple(np.array, np.array)): The indices of the top-k gradient values region reduced.
        """

        # 1. Global topk selection
        coo_reduced_region_global_topk = coo_reduced_region_topk.top_selection(global_th, inplace=False)

        # 2. Data packaging
        # TODO

        # 3. Data balancing
        # TODO

        # 4. Allgatherv using recursive doubling
        coo_allgather_topk = self._allgather(coo_reduced_region_global_topk)
        return coo_allgather_topk, coo_reduced_region_global_topk.get_indexes()

    def _intersect_indexes(self, local_indexes, global_indexes):
        """
        Calculates the intersection of two sets of indices of 2D.
        The assertion statement is only executed when the script is not run in optimized mode (python3 -O script.py).
        Remember that '_has_canonical_format' should only be used for debugging/development purposes to assert that indexes are correct.
        Indexes in scipy are usually in canonical format, so it should not be necessary to evaluate the indexes format.
        When optimized mode is enabled (python3 -O script.py), the assert sentences are not computed.

        Parameters:
            local_indexes (tuple(np.array, np.array)): a tuple of two numpy arrays representing row and column indices, sorted by rows, then by columns.
            global_indexes (tuple(np.array, np.array)): a tuple of two numpy arrays representing row and column indices, sorted by rows, then by columns.

        Returns:
            intersected_indexes (tuple(np.array, np.array)): Set of tuples representing the common indices.

        Example:
            - local_indexes  = (np.array([0, 1, 2, 3, 3, 4]) , np.array([4, 6, 5, 1, 7, 3]))
            - global_indexes = (np.array([0, 1, 3, 3, 3]), np.array([1, 6, 1, 5, 7]))
            - output: (array([1, 3, 3]), array([6, 1, 7]))
        """

        local_rows, local_cols = local_indexes
        global_rows, global_cols = global_indexes
        return intersect_2d_indexes_cython(local_rows, local_cols, global_rows, global_cols)

    def _reduce_topk(self, coo_topk, boundaries, method="p2p_region_wise_reduce_destination_rotation_and_bucketing"):
        """
        Reduce the topk elements in regions defined by boundaries.

        Parameters:
            coo_topk (SparseMatrixCOO): a 2D sparse array in COO format with the values and indexes of topk.
            boundaries (np.array): boundaries for partitioning the gradient space like [row_end_p0, row_end_p1, row_end_p2, ...].
            method (str, optional): The method to use for reduce topk

        Returns:
            coo_reduced_region (SparseMatrixCOO): The reduced topk values in COO format.
        """
        self._show_message_only_once(f"In '_reduce_topk', the method that it is being used is '{method}'")

        if self.model.nprocs == 1:
            return coo_topk

        if method == "collective_allreduce_then_slice":
            warnings.warn("This reduce_topk method ('collective_allreduce_then_slice') should be used only in case of debugging for performance reasons.", RuntimeWarning)
            all_reduced_coo = self.model.comm.allreduce(coo_topk, op=MPI.SUM)
            row_start = 0 if self.model.rank == 0 else boundaries[self.model.rank - 1]
            row_end = boundaries[self.model.rank]
            return all_reduced_coo.slice(row_start, row_end)

        if method == "collective_region_wise_reduce_sync":
            row_start = 0
            reduced_regions_coo = [None] * self.model.nprocs
            for region in range(self.model.nprocs):
                row_end = boundaries[region]
                reduced_regions_coo[region] = self.model.comm.reduce(coo_topk.slice(row_start, row_end), op=MPI.SUM, root=region)
                row_start = row_end
            return reduced_regions_coo[self.model.rank]

        if method == "collective_region_wise_reduce_async":
            """It is not possible with the current mpi4py version to generate a buffer with indexes and values and operate with them"""
            pass

        if method == "p2p_region_wise_reduce_static_destination":
            # Prepare a vector region for storing the partial sums
            coo_region_partial_sum = [None] * self.model.nprocs
            for region in range(self.model.nprocs):
                row_start = 0 if region == 0 else boundaries[region - 1]
                row_end = boundaries[region]
                coo_region_partial_sum[region] = coo_topk.slice(row_start, row_end)

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
                coo_region_partial_sum[region_to_recv] += self.model.comm.sendrecv(coo_region_partial_sum[region_to_send],
                                                                             dest=destination, source=receive_from)

            return coo_region_partial_sum[self.model.rank]

        if method == "p2p_region_wise_reduce_destination_rotation_and_bucketing":
            # There are (nprocs - 1) messages to send (excluding self)
            total_sends = self.model.nprocs - 1
            requests = [None] * total_sends

            # Compute local slice of coo_topk (the "self" region)
            row_start = 0 if self.model.rank == 0 else boundaries[self.model.rank - 1]
            row_end = boundaries[self.model.rank]
            coo_reduced_region = coo_topk.slice(row_start, row_end)

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
                    requests[comm_step + i] = self.model.comm.isend(coo_topk.slice(row_start, row_end), dest=region)
                    region = (region + 1) % self.model.nprocs
                # After sending the bucket, perform the receives sequentially for the same bucket.
                for i in range(current_bucket_size):
                    coo_reduced_region += self.model.comm.recv()

            MPI.Request.Waitall(requests)
            return coo_reduced_region

        raise NotImplementedError(f"Method '{method}' not implemented")

    def _allgather(self, local_data, input_format="SparseMatrixCOO"):
        """
        Gathers data from all processes.

        Parameters:
            local_data (np.ndarray or SparseMatrixCOO): The local data to be gathered.
            input_format (str, optional): The format of the input data.
        Returns:
            gathered_data (np.ndarray or SparseMatrixCOO): The gathered global data in the specified format.
        """

        if self.model.nprocs == 1:
            return local_data

        if input_format == "SparseMatrixCOO":
            gathered = self.model.comm.allgather(local_data.get_triplet())
            all_val = np.concatenate([t[0] for t in gathered])
            all_row = np.concatenate([t[1] for t in gathered])
            all_col = np.concatenate([t[2] for t in gathered])
            return SparseMatrixCOO(all_val, all_row, all_col, self.dw_2d_shape, has_canonical_format=True)

        if input_format == "dense":
            warnings.warn("Try to avoid dense communications!", RuntimeWarning)
            return np.concatenate(self.model.comm.allgather(local_data))

        raise NotImplementedError(f"Input format '{input_format}' not implemented")

    def _show_message_only_once(self, message):
        """
        Show information messages only once to assess the selected functions are being used.

        Parameters:
            message (str): The message to show.
        Returns:
            void (None):
        """
        if self.model.rank == 0:
            if message not in self.info_messages:
                self.info_messages.add(message)
                print(message)

    def _has_canonical_format(self, indexes):
        """
        Check if indexes follows the COO canonical format:
            - Indexes are sorted by row and then by column
            - There are no duplicate entries
        This function is computationally expensive and therefore should only be used for developing/debugging purposes.
        This function should only be used in developement to assert that sparse matrices have canonical format.

        Parameters:
            indexes (tuple(np.ndarray, np.ndarray)): indexes to check

        Returns:
            has_canonical_format (bool): True if indexes are in canonical format, False if not.
        """

        warnings.warn("This function ('has_canonical_format') should be used only in case of debugging for performance reasons.", RuntimeWarning)

        row, col = indexes

        if len(row) != len(col):
            raise AssertionError("'row' and 'col' must have the same length")

        if len(row) == 0:
            return True

        if not np.all(row[:-1] <= row[1:]):
            return False

        for i in range(len(row) - 1):
            if row[i] == row[i + 1] and col[i] >= col[i + 1]:
                return False
        return True
