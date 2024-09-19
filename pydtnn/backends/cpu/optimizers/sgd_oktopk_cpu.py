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
from scipy.sparse import coo_array, csr_array

from pydtnn.cython_modules import \
    intersect_2d_indexes_cython, \
    top_threshold_selection_cython, \
    top_threshold_selection_coo_cython, \
    update_sparsed_weights_cython, \
    update_dense_weights_cython
from pydtnn.backends.cpu.optimizers import OptimizerCPU
from pydtnn.optimizers import SGD_OkTopk

try:
    # noinspection PyUnresolvedReferences,PyPackageRequirements
    from mpi4py import MPI
except (ImportError, ModuleNotFoundError):
    pass



def custom_reduce(local, remote, datatype):
    local_topk, (local_row, local_col) = local
    remote_topk, (remote_row, remote_col) = remote
    
    if len(local_topk) == 0:
        return remote

    if len(remote_topk) == 0:
        return local

    result_dict = {}
    for value, row, col in zip(local_topk, local_row, local_col):
        result_dict[(row, col)] = value
            
    for value, row, col in zip(remote_topk, remote_row, remote_col):
        if (row, col) in result_dict:
            result_dict[(row, col)] += value
        else:
            result_dict[(row, col)] = value
    
    sum_topk = np.array(list(result_dict.values()))
    indices = list(result_dict.keys())
    row, col = zip(*indices)  
    row = np.array(row)
    col = np.array(col)

    return (sum_topk, (row, col))


def custom_numpy_reduce(local, remote, datatype):
    local_topk, (local_row, local_col) = local
    remote_topk, (remote_row, remote_col) = remote

    if len(local_topk) == 0:
        return remote

    if len(remote_topk) == 0:
        return local

    local_matrix = csr_array((local_topk, (local_row, local_col)))
    remote_matrix = csr_array((remote_topk, (remote_row, remote_col)))
    sum_matrix = (local_matrix + remote_matrix).tocoo()
    return (sum_matrix.data, (sum_matrix.row, sum_matrix.col))


op_numpy_reduce = MPI.Op.Create(custom_numpy_reduce, commute=True)
op_custom_reduce = MPI.Op.Create(custom_reduce, commute=True)



class SGD_OkTopkCPU(OptimizerCPU, SGD_OkTopk):


    def update(self, layer, **kwargs):
        current_batch = kwargs.get("current_batch", None)

        if current_batch == 0:
            self.all_local_th[layer.id] = {dw_: None for dw_ in layer.grad_vars.values()}
            self.all_global_th[layer.id] = {dw_: None for dw_ in layer.grad_vars.values()}
            self.all_residuals[layer.id] = {dw_: None for dw_ in layer.grad_vars.values()}
            self.all_boundaries[layer.id] = {dw_: None for dw_ in layer.grad_vars.values()}

        for w_, dw_ in layer.grad_vars.items():
            # Get layer weights and gradients
            w, dw = getattr(layer, w_), getattr(layer, dw_)

            # Initialize current layer-parameter values
            self.dw_shape = dw.shape
            self.local_th = self.all_local_th[layer.id][dw_]
            self.global_th = self.all_global_th[layer.id][dw_]
            self.boundaries = self.all_boundaries[layer.id][dw_]
            if self.all_residuals[layer.id][dw_] is None:
                self.all_residuals[layer.id][dw_] = np.zeros_like(w, dtype=layer.model.dtype)
                
            # Compute acc 
            acc = self.all_residuals[layer.id][dw_] + (self.learning_rate * dw)
            
            # Reshape acc to 2D matrix 
            if len(self.dw_shape) != 2:
                acc = acc.reshape(acc.shape[0], -1)
            self.acc_shape = acc.shape

            # Main part of ok-topk: compute the values that contribute to the update and its indexes
            u, indexes = self._ok_sparse_allreduce(acc, current_batch, self.k, self.tau, self.tau_prime)
               
            # Update residuals
            self.all_residuals[layer.id][dw_] = self._reset_residuals(acc, indexes)
            
            # self.local_th and self.global_th are inmutable, so we have to set them in the dictionary
            self.all_local_th[layer.id][dw_] = self.local_th
            self.all_global_th[layer.id][dw_] = self.global_th

            # Perform the weights update
            self._update_weights(layer, w_, w, u)


    def _reset_residuals(self, acc, indexes):
        """
        Update residuals: set zero value if it is in indexes, else acc value is set.
        
        """

        residuals = np.array(acc)
        if len(indexes[0]) > 0:
            residuals[indexes] = 0
        
        if len(self.dw_shape) != 2:
            residuals = residuals.reshape(self.dw_shape)

        return residuals

    
    def _update_weights(self, layer, w_, w, u, u_format="coo", method="cython"):
        """
        Update weights

        w -= (u / self.nprocs)
        setattr(layer, w_, w)
        """

        if u_format == "dense" and method == "cython": # 3
            u = coo_array(u, shape=self.acc_shape).todense()
            if len(self.dw_shape) != 2:
                w = w.reshape(w.shape[0], -1)
            w = update_dense_weights_cython(w, u, self.nprocs)
            if len(self.dw_shape) != 2:
                w = w.reshape(self.dw_shape)
            setattr(layer, w_, w)  

        elif u_format == "dense" and method == "numpy": # 4
            u = coo_array(u, shape=self.acc_shape).todense()
            if len(self.dw_shape) != 2:
                u = u.reshape(self.dw_shape)
            w -= (u / self.nprocs)
            setattr(layer, w_, w)  

        elif u_format == "coo" and method == "cython": # 2
            grads, (rows, cols) = u
            if len(self.dw_shape) != 2:
                w = w.reshape(w.shape[0], -1)
            w = update_sparsed_weights_cython(w, grads, rows, cols, self.nprocs)
            if len(self.dw_shape) != 2:
                w = w.reshape(self.dw_shape)
            setattr(layer, w_, w)  

        elif u_format == "coo" and method == "numpy": # 1
            grads, indexes = u
            if len(self.dw_shape) != 2:
                w = w.reshape(w.shape[0], -1)
            w[indexes] -= (grads / self.nprocs)
            if len(self.dw_shape) != 2:
                w = w.reshape(self.dw_shape)
            setattr(layer, w_, w)  


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
            - u: The updated gradient values after sparse allreduce.
            - indexes: The indices of the top-k gradient values that were updated.
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


    def _th_re_evaluate(self, tensor, k, input_format="dense"):
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
        
        if input_format == "dense":
            sorted_tensor = np.sort(np.abs(tensor).flatten())
            threshold = sorted_tensor[max(-k, -len(sorted_tensor))]
            return threshold

        elif input_format == "coo":
            data, (_, _) = tensor
            sorted_data = np.sort(np.abs(data))
            threshold = sorted_data[max(-k, -len(sorted_data))]
            return threshold


    def _space_repartition(self, acc, local_th):
        """
        Returns the boundaries of the regions of the gradient matrix for the split and reduce phase.
        
        Parameters:
            - acc: gradient matrix values
            - local_th: local process gradient threshold

        Returns:
            - boundaries: [(row_start, row_end), ...]
                where row_start is included and row_end is excluded.
        """
    
        boundaries = []
        # TODO

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
            - boundaries: Boundaries for partitioning the gradient space: [(row_start_0, row_end_0), ...] .

        Returns:
            - reduced_topk: The reduced top-k gradient values in COO format.
            - local_topk_indexes: The indices of the top-k gradient values selected locally ([row_0, row_1], [col_0, col_1]).
        """
        
        topk, topk_indexes = self._top_threshold_selection(acc, local_th)
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
            - local_indexes: a tuple of two arrays: row and col. 
            - global_indexes: a tuple of two arrays: row and col. 
        
        Returns:
            - Set of tuples representing the common indices.
        
        Example:
            - local_indexes = (np.array([0, 3, 2]), np.array([2, 4, 1]))
            - global_indexes = (np.array([1, 2, 3]), np.array([3, 1, 4]))
            - output: (array([3, 2]), array([4, 1]))
        """

        if method == "cython":
            local_rows, local_cols = local_indexes
            global_rows, global_cols = global_indexes
            local_rows = np.array(local_rows, dtype=np.int32)
            local_cols = np.array(local_cols, dtype=np.int32)
            global_rows = np.array(global_rows, dtype=np.int32)
            global_cols = np.array(global_cols, dtype=np.int32)
            return intersect_2d_indexes_cython(local_rows, local_cols, global_rows, global_cols)
        
        if method == "numpy":
            local_row, local_col = local_indexes
            global_row, global_col = global_indexes
            intersected_indexes = np.array([]), np.array([])

            local_tuples = set(zip(local_row, local_col))
            global_tuples = set(zip(global_row, global_col))
            intersected_tuples = local_tuples.intersection(global_tuples)

            if intersected_tuples:
                intersected_row, intersected_col = zip(*intersected_tuples)
                intersected_indexes = np.array(intersected_row), np.array(intersected_col)
                
            return intersected_indexes
          

    def _top_threshold_selection(self, matrix, threshold, input_format="dense", method="cython"):
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

        elif input_format == "coo" and method == "cython":
            data, (row, col) = matrix
            topk, topk_indexes = top_threshold_selection_coo_cython(data, row, col, threshold)
            return topk, topk_indexes

        elif input_format == "dense":
            topk_indexes = np.where(np.abs(matrix) >= threshold)
            topk = matrix[topk_indexes]
            return topk, topk_indexes

        elif input_format == "coo":
            data, (row, col) = matrix
            mask = np.abs(data) >= threshold
            topk = data[mask]
            topk_indexes = (row[mask], col[mask])
            return topk, topk_indexes


    def _reduce_topk(self, topk, topk_indexes, boundaries):
        """
        TODO: To simplify the implementation, destination rotation and bucketing,
        is not yet implemented.

        TODO: Boundaries are not being used
        """

        if self.nprocs == 1:
            return topk, topk_indexes

        reduced_topk, reduced_indexes = None, None
        topk_splitted = np.array_split(topk, self.nprocs)
        row_splitted = np.array_split(topk_indexes[0], self.nprocs) 
        col_splitted = np.array_split(topk_indexes[1], self.nprocs)
        
        for region in range(self.nprocs):
            region_topk = (topk_splitted[region], (row_splitted[region], col_splitted[region]))
            if self.rank == region:
                reduced_topk, reduced_indexes = self.comm.reduce(region_topk, op=op_custom_reduce, root=region)
            else:
                _ = self.comm.reduce(region_topk, op=op_custom_reduce, root=region)
        return reduced_topk, reduced_indexes


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

