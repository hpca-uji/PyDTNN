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

import warnings
import numpy as np
from scipy.sparse import coo_array

from pydtnn.cython_modules import \
    compute_dense_acc_cython, \
    intersect_2d_indexes_cython, \
    top_threshold_selection_cython, \
    top_threshold_selection_coo_cython, \
    update_sparsed_weights_cython, \
    update_sparsed_weights_mv_cython, \
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

        if self.density == 1:
            return np.zeros_like(acc)

        if method == "cython":
            assert(self._has_canonical_format(indexes))
            return reset_residuals_cython(acc, indexes[0], indexes[1])
        
        if method == "numpy":
            if len(indexes[0]) > 0:
                acc[indexes] = 0    
            return acc

        raise NotImplementedError(f"Method '{method}' not implemented")

    
    def _update_weights(self, layer, w_type, w, coo_u, method="cython_with_vel_and_momentum"):
        """
        Update weights: w -= (u / self.nprocs) and set to weight layer attribute: setattr(layer, w_type, w)

        Parameters:
            layer (int): layer id
            w_type (string): weight param type (bias, weight, ...)
            w (np.array): N dimensional dense weights matrix/tensor 
            coo_u (coo_array): Sparse 2D gradient matrix in COO format to update w
            method (string, optional): The method to use for updating the weights. It can be 'cython' or 'numpy'. Default is 'cython'.

        Returns:
            (void): instead it directly applies the result to the weight layer attribute
        """

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
                warnings.warn("If momentum is 0 use 'cython' method, it produces the same output but it is faster")

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
                warnings.warn("If momentum is 0 use just 'numpy' method, it produces the same output but it is faster")

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
            warnings.warn("This method should be used only in case of debugging for performance reasons.")

            dw = coo_u.toarray()
            if len(self.dw_original_shape) != 2:
                dw = dw.reshape(self.dw_original_shape)
            velocity = getattr(layer, "velocity_%s" % w_type, np.zeros_like(w, dtype=layer.model.dtype))
            velocity = self.momentum * velocity + dw
            w -= velocity # Oktopk already computes acc with learning_rate
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
                - coo_u (coo_array): The updated gradient values in 2D sparse format.
                - indexes (tuple(np.array, np.array)): The indices of the top-k gradient values that were updated.
        """

        if t % thresholds_re_evaluation_t == 0:
            self.local_th = self._th_re_evaluate(acc, k, input_format="dense")
        
        if t % space_repartition_t == 0:
            self.boundaries = self._space_repartition(acc, self.local_th)

        coo_reduced_region_topk, local_topk_indexes = self._split_and_reduce(acc, self.local_th, self.boundaries)
        
        if t % thresholds_re_evaluation_t == 0:
            coo_all_reduced_topk = self._allgather(coo_reduced_region_topk, input_format="coo")
            self.global_th = self._th_re_evaluate(coo_all_reduced_topk, k, input_format="coo")

        coo_u, global_topk_indexes = self._balance_and_allgather(coo_reduced_region_topk, self.global_th)
        indexes = self._intersect_indexes(local_topk_indexes, global_topk_indexes)
        return coo_u, indexes


    def _th_re_evaluate(self, matrix, k, input_format=None, method="numpy_sort"):
        """
        Return the absolute gradient threshold for a given matrix.
        
        Parameters:
            matrix (np.array or coo_array): A 2D gradient matrix, in np.array for 'dense' input_format or 
                                            coo_array for 'coo' input_format.
            k (int): Indicating the number of top gradient values to consider.
            input_format (string): Either 'dense' for a dense matrix or 'coo' for a sparse matrix in COO format.
            method (string, optional): The method to use for threshold selection. It can be 'numpy_sort' or 'numpy_partition'.
        
        Returns:
            threshold (float): The absolute gradient threshold based on the top k values.
        """
        
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


    def _space_repartition(self, acc, local_th, balanced=False):
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
    
        if not balanced:
            boundaries = np.zeros(self.nprocs, dtype=np.int32)
            total_rows = self.dw_original_shape[0]
            block_size = total_rows // self.nprocs
            for i in range(0, self.nprocs - 1):
                boundaries[i] = block_size * (i + 1)
            boundaries[self.nprocs - 1] = total_rows
            return boundaries

        if balanced:
            coo_topk = self._top_threshold_selection(acc, local_th, input_format="dense")
            
            current_row = 0
            current_proc = 0
            rows = coo_topk.row 
            topk_in_current_proc = 0
            total_rows = coo_topk.shape[0]
            boundaries = np.zeros(self.nprocs, dtype=np.int32)
            topk_per_proc = coo_topk.count_nonzero() // self.nprocs
            topk_per_row = np.zeros(total_rows, dtype=np.int32)
            np.add.at(topk_per_row, rows, 1)

            while current_proc < self.nprocs - 1:
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
            boundaries[self.nprocs - 1] = total_rows

            global_boundaries = self.comm.allreduce(boundaries, op=MPI.SUM) // self.nprocs
            
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
                - coo_reduced_region_topk (coo_array): The reduced top-k gradient values in COO format.
                - local_topk_indexes (tuple(np.array, np.array)): The indices of the top-k gradient values selected locally.
        """
        
        coo_topk = self._top_threshold_selection(acc, local_th, input_format="dense")
        coo_reduced_region_topk = self._reduce_topk(coo_topk, boundaries)
        return coo_reduced_region_topk, (coo_topk.row, coo_topk.col)


    def _balance_and_allgather(self, coo_reduced_region_topk, global_th):
        """
        Second main phase of ok_sparse_allreduce.  
        Performs the allgather of the coo_reduced_region_topk values among workers.

        Parameters:
            coo_reduced_region_topk (coo_array): a 2D sparse gradient matrix.
            global_th (float): the global threshold to perfrom top selection.

        Returns:
            out (tuple with two elements:):
                - coo_allgather_topk (coo_array): A 2D sparse gradient matrix with the global top-k selection.
                - reduced_region_global_topk_indexes (tuple(np.array, np.array)): The indices of the top-k gradient values region reduced.
        """

        # 1. Global topk selection
        coo_reduced_region_global_topk = self._top_threshold_selection(coo_reduced_region_topk, global_th, input_format="coo")

        # 2. Data packaging
        # TODO

        # 3. Data balancing
        # TODO

        # 4. Allgatherv using recursive doubling
        coo_allgather_topk = self._allgather(coo_reduced_region_global_topk, input_format="coo")
        return coo_allgather_topk, (coo_reduced_region_global_topk.row, coo_reduced_region_global_topk.col) 


    def _has_canonical_format(self, indexes):
        """
        Check if indexes are sorted by row and then by column (COO canonical format)
        This function is computationally expensive and therefore should only be used for developing/debugging purposes.
        This function should only be used in developement to assert that sparse matrices have canonical format. 

        Parameters:
            indexes (tuple(np.array, np.array)): a tuple of rows and cols

        Returns:
            has_canonical_format (boolean): True if indexes are in canonical format, False if not. 
        """
        
        warnings.warn("This function ('_has_canonical_format') should be used only in case of debugging for performance reasons.")
        
        rows, cols = indexes

        if len(rows) == 0:
            return True

        if not np.all(rows[:-1] <= rows[1:]):
            return False

        for i in range(len(rows) - 1):
            if rows[i] == rows[i + 1] and cols[i] > cols[i + 1]:
                return False
        return True


    def _intersect_indexes(self, local_indexes, global_indexes, method="cython"):
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

        assert(self._has_canonical_format(local_indexes) and self._has_canonical_format(global_indexes)) 

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
            matrix (np.array o coo_array): The input 2D matrix from which to select elements.
            threshold (float): The threshold value to compare against the absolute values of the matrix elements.
            input_format (str): The format of the input matrix. It can be 'dense' for numpy arrays or 'coo' for sparse COO arrays.
            method (str, optional): The method to use for selection. It can be "cython" or "numpy". Default is "cython".
        
        Returns:
            topk (coo_array): A sparse 2D gradient matrix topk in COO.
        """

        if method == "cython" and input_format == "dense":
            topk, topk_indexes = top_threshold_selection_cython(matrix, threshold)
            return coo_array((topk, topk_indexes), dtype=np.float32, shape=self.dw_2d_shape)

        if method == "cython" and input_format == "coo":
            topk, topk_indexes = top_threshold_selection_coo_cython(matrix.data, matrix.row, matrix.col, threshold)
            return coo_array((topk, topk_indexes), dtype=np.float32, shape=self.dw_2d_shape)

        if method == "numpy" and input_format == "dense":
            topk_indexes = np.where(np.abs(matrix) >= threshold)
            topk = matrix[topk_indexes]
            return coo_array((topk, topk_indexes), dtype=np.float32, shape=self.dw_2d_shape)

        if method == "numpy" and input_format == "coo":
            mask = np.abs(matrix.data) >= threshold
            topk = matrix.data[mask]
            topk_indexes = (matrix.row[mask], matrix.col[mask])
            return coo_array((topk, topk_indexes), dtype=np.float32, shape=self.dw_2d_shape)

        raise NotImplementedError(f"Method '{method}' with format '{input_format}' not implemented")


    def _reduce_topk(self, coo_topk, boundaries, method="reduce_region"):
        """
        Reduce the topk elements in regions defined by boundaries.

        Parameters:
            coo_topk (coo_array): a 2D sparse array in COO format with the values and indexes of topk.
            boundaries (np.array): boundaries for partitioning the gradient space like [row_end_p0, row_end_p1, row_end_p2, ...].

        Warning:
            Method 'reduce_region' does not provide the same exact accuracy as 'reduce_region_blocking' or 'allreduce_then_slice'.

        Returns:
            coo_reduced_region (coo_array): The reduced topk values in COO format.
        """

        if self.nprocs == 1:
            return coo_topk

        if method == "allreduce_then_slice":
            all_reduced_csr = self.comm.allreduce(coo_topk, op=MPI.SUM)
            row_start = 0 if self.rank == 0 else boundaries[self.rank - 1]
            row_end = boundaries[self.rank]
            coo_reduced_region = all_reduced_csr[row_start:row_end].tocoo()
            coo_reduced_region.row += row_start
            return coo_reduced_region

        if method == "reduce_region_blocking":
            row_start = 0
            csr_topk = coo_topk.tocsr()
            reduced_regions_csr = [None] * self.nprocs
            for region in range(self.nprocs):
                row_end = boundaries[region]
                reduced_regions_csr[region] = self.comm.reduce(csr_topk[row_start:row_end], op=MPI.SUM, root=region)
                row_start = row_end
            coo_reduced_region = reduced_regions_csr[self.rank].tocoo()
            if self.rank != 0:
                coo_reduced_region.row += boundaries[self.rank - 1]
            # Another coo_array conversion is needed to set shape as dw_2d_shape 
            coo_reduced_region = coo_array((coo_reduced_region.data, (coo_reduced_region.row, coo_reduced_region.col)), dtype=np.float32, shape=self.dw_2d_shape)
            return coo_reduced_region

        if method == "reduce_region":
            row_start = 0
            requests = [None] * self.nprocs
            recv_bufs = [None] * self.nprocs
            csr_topk = coo_topk.tocsr()
            for region in range(self.nprocs):
                row_end = boundaries[region]
                send_buf = csr_topk[row_start:row_end].toarray()
                recv_bufs[region] = np.zeros_like(send_buf) if self.rank == region else None
                requests[region] = self.comm.Ireduce(send_buf, recv_bufs[region], op=MPI.SUM, root=region)
                row_start = row_end
            MPI.Request.Waitall(requests)
            if recv_bufs[self.rank] is not None:
                coo_reduced_region = coo_array(recv_bufs[self.rank], dtype=np.float32)
                if self.rank != 0:
                    coo_reduced_region.row += boundaries[self.rank - 1]
                # Another coo_array conversion is needed to set shape as dw_2d_shape 
                coo_reduced_region = coo_array((coo_reduced_region.data, (coo_reduced_region.row, coo_reduced_region.col)), dtype=np.float32, shape=self.dw_2d_shape)
                return coo_reduced_region
            return coo_array((np.array([]), (np.array([]), np.array([]))), shape=self.dw_2d_shape, dtype=np.float32)

        raise NotImplementedError(f"Method '{method}' not implemented")


    def _allgather(self, local_data, input_format=None):
        """
        Gathers data from all processes and concatenates it into a single array.
        
        Parameters:
            local_data (numpy.ndarray or coo_array): The local data to be gathered.
            input_format (str): The format of the input data. Supported formats are "coo" for 
                                coordinate format sparse matrices and "dense" for dense arrays.
        Returns:
            gathered_data (numpy.ndarray or coo_array): The gathered global data in the specified format.
        """
        if self.nprocs == 1:
            return local_data
        
        if input_format == "coo":
            local_tuple = (local_data.data, local_data.row, local_data.col)
            gathered = self.comm.allgather(local_tuple)
            all_val = np.concatenate([t[0] for t in gathered])
            all_row = np.concatenate([t[1] for t in gathered])
            all_col = np.concatenate([t[2] for t in gathered])
            coo_gathered_data = coo_array((all_val, (all_row, all_col)), shape=self.dw_2d_shape, dtype=np.float32)
            return coo_gathered_data

        if input_format == "dense":
            return np.concatenate(self.comm.allgather(local_data))

        raise NotImplementedError(f"Input format '{input_format}' not implemented")

