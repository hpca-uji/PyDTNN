#
#  This file is part of Python Distributed Training of Neural Networks (PyDTNN)
#
#  Copyright (C) 2021 Universitat Jaume I
#
#  PyDTNN is free software: you can redistribute it and/or modify it under the
#  terms of the GNU General Public License as published by the Free Software
#  Foundation, either version 3 of the License, or (at your option) any later
#  version.
#
#  This program is distributed in the hope that it will be useful, but WITHOUT
#  ANY WARRANTY; without even the implied warranty of MERCHANTABILITY
#  or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public
#  License for more details.
#
#  You should have received a copy of the GNU General Public License along
#  with this program. If not, see <https://www.gnu.org/licenses/>.
#

import numpy as np
from scipy.sparse import csr_array, coo_array

from pydtnn.cython_modules import \
    compute_dense_acc_cython, \
    intersect_2d_indexes_cython, \
    top_threshold_selection_cython, \
    top_threshold_selection_coo_cython, \
    update_sparsed_weights_cython, \
    update_dense_weights_cython, \
    reset_residuals_cython
from pydtnn.backends.cpu.optimizers import OptimizerCPU
from pydtnn.optimizers import OkTopk

try:
    # noinspection PyUnresolvedReferences,PyPackageRequirements
    from mpi4py import MPI
except (ImportError, ModuleNotFoundError):
    pass



class OkTopkCPU(OptimizerCPU, OkTopk):


    def update(self, layer):
       
        if layer.id not in self.iterations:
            self.iterations[layer.id] = 0
            self.all_local_th[layer.id] = {dw_: None for dw_ in layer.grad_vars.values()}
            self.all_global_th[layer.id] = {dw_: None for dw_ in layer.grad_vars.values()}
            self.all_residuals[layer.id] = {dw_: None for dw_ in layer.grad_vars.values()}
            self.all_boundaries[layer.id] = {dw_: None for dw_ in layer.grad_vars.values()}

        for w_, dw_ in layer.grad_vars.items():
            # Get layer weights and gradients
            w, dw = getattr(layer, w_), getattr(layer, dw_)

            # Reshape dw to 2D matrix
            self.dw_original_shape = dw.shape
            if len(self.dw_original_shape) != 2:
                dw = dw.reshape(dw.shape[0], -1)
            self.dw_2d_shape = dw.shape

            # Initialize current layer-parameter values
            self.local_th = self.all_local_th[layer.id][dw_]
            self.global_th = self.all_global_th[layer.id][dw_]
            self.boundaries = self.all_boundaries[layer.id][dw_]
            if self.all_residuals[layer.id][dw_] is None:
                self.all_residuals[layer.id][dw_] = np.zeros_like(dw, dtype=layer.model.dtype)
                
            # Compute acc 
            acc = self._compute_acc(self.all_residuals[layer.id][dw_], dw, self.learning_rate)

            # Main part of ok-topk: compute the values that contribute to the update and its indexes
            u, indexes = self._ok_sparse_allreduce(acc, self.iterations[layer.id], self.k, self.tau, self.tau_prime)
               
            # Update residuals
            self.all_residuals[layer.id][dw_] = self._reset_residuals(acc, indexes)
            
            # Save for next updates thresholds and boundaries
            self.all_local_th[layer.id][dw_] = self.local_th
            self.all_global_th[layer.id][dw_] = self.global_th
            self.all_boundaries[layer.id][dw_] = self.boundaries

            # Perform the weights update
            self._update_weights(layer, w_, w, u)

        self.iterations[layer.id] += 1


    def _compute_acc(self, residuals, dw, learning_rate, method="cython"):
        """
        Compute acc, where: acc = residuals + (learning_rate * dw)

        Parameters:
            - residuals: 2D dense matrix with the current layer residuals
            - dw: 2D dense matrix with the current layer gradients
            - learning_rate: learning rate float value

        Returns acc = residuals + (learning_rate * dw)
        """

        if method == "cython":
            return compute_dense_acc_cython(residuals, dw, learning_rate)
        
        if method == "numpy":
            return residuals + (learning_rate * dw)

        raise NotImplementedError(f"Method '{method}' not implemented")


    def _reset_residuals(self, acc, indexes, method="cython"):
        """
        Update residuals: set zero value if it is in indexes, else acc value is set

        Parameters:
            - acc: 2D dense matrix
            - indexes: a tuple with two np.arrays (rows, cols)

        Returns:
            - residuals, which is the same as acc with the values in indexes set to zero 
        """

        if method == "cython":
            return reset_residuals_cython(acc, indexes[0], indexes[1])
        
        if method == "numpy":
            if len(indexes[0]) > 0:
                acc[indexes] = 0    
            return acc

        raise NotImplementedError(f"Method '{method}' not implemented")

    
    def _update_weights(self, layer, w_, w, u, method="cython"):
        """
        Update weights: w -= (u / self.nprocs) and set to weight layer attribute: setattr(layer, w_, w)

        Parameters:
            - layer: layer id
            - w_: weight param type (bias, weight, ...)
            - w: N dimensional dense weights matrix/tensor 
            - u: Sparse 2D gradient matrix in coo format to update w

        Returns:
            - void, instead it directly applies the result to the weight layer attribute
        """

        if method == "cython": 
            grads, (rows, cols) = u
            if len(self.dw_original_shape) != 2:
                w = w.reshape(w.shape[0], -1)
            w = update_sparsed_weights_cython(w, grads, rows, cols, self.nprocs)
            if len(self.dw_original_shape) != 2:
                w = w.reshape(self.dw_original_shape)
            setattr(layer, w_, w)  
            return

        if method == "numpy": 
            grads, indexes = u
            if len(self.dw_original_shape) != 2:
                w = w.reshape(w.shape[0], -1)
            w[indexes] -= (grads / self.nprocs)
            if len(self.dw_original_shape) != 2:
                w = w.reshape(self.dw_original_shape)
            setattr(layer, w_, w)  
            return

        raise NotImplementedError(f"Method '{method}' with format '{input_format}' not implemented")


    def _ok_sparse_allreduce(self, acc, t, k, space_repartition_t, thresholds_re_evaluation_t):
        """
        Performs the Ok-Topk sparse allreduce operation. 
        This method executes the Ok-Topk sparse allreduce algorithm, which 
        optimizes communication by only exchanging the most significant 
        gradient values (top-k) across distributed processes. The method 
        periodically re-evaluates the thresholds and repartitions the 
        gradient space to maintain efficiency and accuracy.

        Parameters:
            - acc: Gradient matrix accumulation values.
            - t: Current iteration number.
            - k: Number of top-k gradient values to select.
            - space_repartition_t: Interval of iterations for space repartitioning.
            - thresholds_re_evaluation_t: Interval of iterations for threshold re-evaluation.

        Returns:
            - u: The updated gradient values in coo 2D sparse format.
            - indexes: The indices of the top-k gradient values that were updated: (row, col).
        """

        if t % thresholds_re_evaluation_t == 0:
            self.local_th = self._th_re_evaluate(acc, k, input_format="dense")
        
        if t % space_repartition_t == 0:
            self.boundaries = self._space_repartition(acc, self.local_th)

        reduced_topk, local_topk_indexes = self._split_and_reduce(acc, self.local_th, self.boundaries)
        
        if t % thresholds_re_evaluation_t == 0:
            all_reduced_topk = self._allgather(reduced_topk)
            self.global_th = self._th_re_evaluate(all_reduced_topk, k, input_format="coo")

        u, global_topk_indexes = self._balance_and_allgather(reduced_topk, self.global_th)
        indexes = self._intersect_indexes(local_topk_indexes, global_topk_indexes)
        return u, indexes


    def _th_re_evaluate(self, tensor, k, input_format=None):
        """
        Return the absolute gradient threshold for a given tensor.
        
        Parameters:
            - tensor: A gradient tensor, expected in dense format for 'dense' input_format 
                    or in COO format (data, (row, col)) for 'coo' input_format.
            - k: An integer, indicating the number of top gradient values to consider.
            - input_format: A string, either 'dense' for a dense tensor or 'coo' for a sparse tensor in COO format.
        
        Returns:
            - threshold: The absolute gradient threshold based on the top k values.
        """
        
        if k <= 0:
            return 0.0
        
        if input_format == "coo" and tensor[0].size == 0:
            return 1.0

        if input_format == "dense":
            sorted_tensor = np.sort(np.abs(tensor).flatten())
            threshold = sorted_tensor[max(-k, -len(sorted_tensor))]
            return threshold

        if input_format == "coo":
            data, (_, _) = tensor
            sorted_data = np.sort(np.abs(data))
            threshold = sorted_data[max(-k, -len(sorted_data))]
            return threshold
        
        raise NotImplementedError(f"Input format '{input_format}' not implemented")


    def _space_repartition(self, acc, local_th, balanced=False):
        """
        Returns the boundaries of the regions of the gradient matrix for the split and reduce phase.
        
        Parameters:
            - acc: gradient matrix values
            - local_th: local process gradient threshold
            - balanced: if not balanced, a static row partition is performed, 
                        if balanced, a topk gradiend distribution is considered in the row partition  

        Returns:
            - boundaries: [row_end_p0, row_end_p1, row_end_p2, ...]
        """
    
        if not balanced:
            boundaries = []
            rows = self.dw_original_shape[0]
            block_size = rows // self.nprocs
            for i in range(1, self.nprocs + 1):
                if i == self.nprocs:
                    boundaries.append(rows)
                else:
                    boundaries.append(block_size * i)
            return boundaries
        
        if balanced:
            _, (row, col) = self._top_threshold_selection(acc, local_th, input_format="dense")
            all_topk_row = self._allgather(row, input_format="dense")
            all_topk_col = self._allgather(col, input_format="dense")
            
            indexes_set = set()
            for i in range(len(all_topk_row)):
                indexes_set.add((all_topk_row[i], all_topk_col[i]))

            rows_count = np.zeros(shape=(self.dw_2d_shape[0],), dtype=np.int32)
            for row, _ in indexes_set:
                rows_count[row] += 1
                
            boundaries = []
            topk_counter = 0
            topk_per_worker = np.sum(rows_count) // self.nprocs

            for row, count in enumerate(rows_count):
                if count > 0:
                    topk_counter += count
                
                while topk_counter >= topk_per_worker:
                    boundaries.append(row + 1)
                    topk_counter -= topk_per_worker  

            if not boundaries or boundaries[-1] != len(rows_count):
                boundaries.append(len(rows_count))

            return boundaries


    def _split_and_reduce(self, acc, local_th, boundaries):
        """
        First main phase of ok_sparse_allreduce.  
        Split the gradients into partitions and reduce them by selecting top-k values.
        Each worker receives sparse regions from the other workers and and then conducts
        the reduction locally. 

        Parameters:
            - acc: 2D gradient matrix accumulation values (in dense format).
            - local_th: Local threshold for selecting top-k values.
            - boundaries: Boundaries for partitioning the gradient space: [row_end_p0, row_end_p1, row_end_p2, ...]

        Returns:
            - reduced_topk: The reduced top-k gradient values in COO format.
            - local_topk_indexes: The indices of the top-k gradient values selected locally ([row_0, row_1], [col_0, col_1]).
        """
        
        topk, topk_indexes = self._top_threshold_selection(acc, local_th, input_format="dense")
        reduced_topk = self._reduce_topk(topk, topk_indexes, boundaries)
        return reduced_topk, topk_indexes


    def _balance_and_allgather(self, reduced_topk, global_th):
        """
        Second main phase of ok_sparse_allreduce.  
        Performs the allgather of the reduced_topk values among workers.

        Parameters:
            - reduced_topk: a sparse gradient matrix (not in coo format)
            - global_th: the global threshold (float) to perfrom top selection

        Returns:
            - global_topk: a sparse gradient matrix with the global topk selection (not in coo format)
            - global_topk_indexes: the indices of the top-k gradient values reduced
        """

        # 1. Global topk selection
        global_topk, global_topk_indexes = self._top_threshold_selection(reduced_topk, global_th, input_format="coo")

        # 2. Data packaging
        # TODO

        # 3. Data balancing
        # TODO

        # 4. Allgatherv using recursive doubling
        allgather_topk, allgather_indexes = self._allgather((global_topk, global_topk_indexes))
        return (allgather_topk, allgather_indexes), global_topk_indexes


    def _intersect_indexes(self, local_indexes, global_indexes, method="cython"):
        """
        Calculates the intersection of two sets of indices of 2D.

        Parameters:
            - local_indexes: a tuple of two arrays: row and col (Sort by rows, then columns). 
            - global_indexes: a tuple of two arrays: row and col (Sort by rows, then columns). 
        
        Returns:
            - Set of tuples representing the common indices.
        
        Example:
            - local_indexes  = (np.array([0, 1, 2, 3, 3, 4]) , np.array([4, 6, 5, 1, 7, 3]))
            - global_indexes = (np.array([0, 1, 3, 3, 3]), np.array([1, 6, 1, 5, 7]))
            - output: (array([1, 3, 3]), array([6, 1, 7]))  
        """

        if method == "cython":
            local_rows, local_cols = local_indexes
            global_rows, global_cols = global_indexes
            return intersect_2d_indexes_cython(local_rows, local_cols, global_rows, global_cols)
        
        if method == "numpy":
            count = 0
            i_local, i_global = 0, 0
            local_rows, local_cols = local_indexes
            global_rows, global_cols = global_indexes
            max_size = min(len(local_rows), len(global_rows))
            intersected_rows = np.empty(max_size, dtype=np.int32)
            intersected_cols = np.empty(max_size, dtype=np.int32)
            while i_local < len(local_rows) and i_global < len(global_rows):
                local_row = local_rows[i_local]
                global_row = global_rows[i_global]
                if local_row < global_row:
                    i_local += 1
                elif local_row > global_row:
                    i_global += 1
                else:
                    local_col = local_cols[i_local]
                    global_col = global_cols[i_global]
                    if local_col < global_col:
                        i_local += 1
                    elif local_col > global_col:
                        i_global += 1
                    else:
                        intersected_rows[count] = local_row
                        intersected_cols[count] = local_col
                        i_global += 1                    
                        i_local += 1
                        count += 1
            return intersected_rows[:count], intersected_cols[:count]

        raise NotImplementedError(f"Method '{method}' not implemented")


    def _top_threshold_selection(self, matrix, threshold, input_format=None, method="cython"):
        """
        Selects top-k elements from the matrix that are greater than or equal to the threshold.
        
        Parameters:
            - matrix (np.array): The input 2D matrix from which to select elements.
            - threshold (float): The threshold value to compare against the absolute values of the matrix elements.
        
        Returns:
            - topk (np.array): A np.array with only the elements that meet the threshold condition 
            - topk_indexes (tuple(np.array, np.array): a tuple of np.arrays with the indexes, e.g (array([0, 1]), array([1, 1]))
        """

        if input_format == "dense" and method == "cython":
            topk, topk_indexes = top_threshold_selection_cython(matrix, threshold)
            return topk, topk_indexes

        if input_format == "coo" and method == "cython":
            data, (row, col) = matrix
            topk, topk_indexes = top_threshold_selection_coo_cython(data, row, col, threshold)
            return topk, topk_indexes

        if input_format == "dense" and method == "numpy":
            topk_indexes = np.where(np.abs(matrix) >= threshold)
            topk = matrix[topk_indexes]
            return topk, topk_indexes

        if input_format == "coo" and method == "numpy":
            data, (row, col) = matrix
            mask = np.abs(data) >= threshold
            topk = data[mask]
            topk_indexes = (row[mask], col[mask])
            return topk, topk_indexes

        raise NotImplementedError(f"Method '{method}' with format '{input_format}' not implemented")


    def _reduce_topk(self, topk, topk_indexes, boundaries, method="reduce_region"):
        """
        Reduce the topk elements in regions defined by boundaries

        Parameters:
            - topk: a np.array with the values of the topk
            - topk_indexes: a tuple of two np.array with the row and col of topk indexes
            - boundaries: boundaries for partitioning the gradient space: [row_end_p0, row_end_p1, row_end_p2, ...]

        Returns:
            - The reduced topk values in coo format: (data, (row, col))
        """

        if self.nprocs == 1:
            return topk, topk_indexes

        if method == "allreduce":
            row_start = 0
            row_end = boundaries[self.rank]
            if self.rank != 0:
                row_start = boundaries[self.rank - 1]
            coo_topk = coo_array((topk, topk_indexes), shape=self.dw_2d_shape, dtype=self.dtype)
            all_reduced_csr = self.comm.allreduce(coo_topk, op=MPI.SUM)
            coo_region = all_reduced_csr[row_start:row_end].tocoo()
            row = coo_region.row + row_start
            return coo_region.data, (row, coo_region.col)

        if method == "reduce_region_blocking":
            row_start = 0
            csr_topk = coo_array((topk, topk_indexes), shape=self.dw_2d_shape, dtype=self.dtype).tocsr()
            reduced_regions_csr = []
            for region in range(self.nprocs):
                row_end = boundaries[region]
                reduced_regions_csr.append(self.comm.reduce(csr_topk[row_start:row_end], op=MPI.SUM, root=region))
                row_start = row_end
            reduced_region_coo = reduced_regions_csr[self.rank].tocoo()
            if self.rank != 0:
                reduced_region_coo.row += boundaries[self.rank - 1]
            return reduced_region_coo.data, (reduced_region_coo.row, reduced_region_coo.col)

        if method == "reduce_region":
            requests = []
            row_start = 0
            recv_bufs = [None] * self.nprocs
            csr_topk = coo_array((topk, topk_indexes), shape=self.dw_2d_shape, dtype=self.dtype).tocsr()
            for region in range(self.nprocs):
                row_end = boundaries[region]
                send_buf = csr_topk[row_start:row_end].toarray()
                if self.rank == region:
                    recv_bufs[region] = np.zeros_like(send_buf)
                requests.append(self.comm.Ireduce(send_buf, recv_bufs[region], op=MPI.SUM, root=region))
                row_start = row_end
            MPI.Request.Waitall(requests)
            if recv_bufs[self.rank] is not None:
                reduced_region_coo = csr_array(recv_bufs[self.rank]).tocoo()
                if self.rank != 0:
                    reduced_region_coo.row += boundaries[self.rank - 1]
                return reduced_region_coo.data, (reduced_region_coo.row, reduced_region_coo.col)
            return None, (None, None)

        raise NotImplementedError(f"Method '{method}' not implemented")


    def _allgather(self, data, input_format="coo"):
        if self.nprocs == 1:
            return data
        
        if input_format == "coo":
            val, (row, col) = data
            all_val = np.concatenate(self.comm.allgather(val))
            all_row = np.concatenate(self.comm.allgather(row))
            all_col = np.concatenate(self.comm.allgather(col))
            return all_val, (all_row, all_col)

        if input_format == "dense":
            return np.concatenate(self.comm.allgather(data))

        raise NotImplementedError(f"Input format '{input_format}' not implemented")

