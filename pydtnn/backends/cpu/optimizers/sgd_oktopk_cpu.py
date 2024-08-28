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
from scipy.sparse import coo_matrix

from pydtnn.cython_modules import top_threshold_selection_cython, flattened_top_threshold_selection_cython
from pydtnn.backends.cpu.optimizers import OptimizerCPU
from pydtnn.optimizers import SGD_OkTopk

try:
    # noinspection PyUnresolvedReferences,PyPackageRequirements
    from mpi4py import MPI
except (ImportError, ModuleNotFoundError):
    pass

class SGD_OkTopkCPU(OptimizerCPU, SGD_OkTopk):


    def update(self, layer, **kwargs):
        current_batch = kwargs.get("current_batch", None)

        if layer.id not in self.residuals:
            self.residuals[layer.id] = {dw_: None for dw_ in layer.grad_vars.values()}

        # TODO: This variable and the if else in the for, should be removed. Use only for testing.
        #       "weights", "biases", "beta", "gamma" should always be trained using oktopk, not SGD.
        
        oktopk_trainable_params = ["weights"] 
         
        for w_, dw_ in layer.grad_vars.items():
            if w_ in oktopk_trainable_params:
                w, dw = getattr(layer, w_), getattr(layer, dw_)
                if self.residuals[layer.id][dw_] is None:
                    self.residuals[layer.id][dw_] = np.zeros_like(w, dtype=layer.model.dtype)

                acc = self.residuals[layer.id][dw_] + (self.learning_rate * dw)
                u, indexes = self._ok_sparse_allreduce(acc, current_batch, self.k)
                self.residuals[layer.id][dw_] = self._update_residuals(acc, indexes)
                w = w - u / self.nprocs
                # w[indexes] = u[indexes] / self.nprocs
                # w[indexes] = u / self.nprocs
                setattr(layer, w_, w)
            else:
                lr = self.learning_rate
                w, dw = getattr(layer, w_), getattr(layer, dw_)
                velocity = getattr(layer, "velocity_%s" % w_, np.zeros_like(w, dtype=layer.model.dtype))

                velocity = self.momentum * velocity + dw
                if self.nesterov:
                    w -= lr * (self.decay * w + dw + self.momentum * velocity)
                else:
                    w -= lr * (self.decay * w + velocity)

                setattr(layer, w_, w)
                setattr(layer, "velocity_%s" % w_, velocity)


    def _update_residuals(self, acc, indexes, method="numpy"):
        """
        Returns the residuals: set zero value if it is in indexes, else acc value is set.

        Parameters:
            - acc: gradient matrix accumulation values (can be multi-dimensional)
            - indexes: set of tuples representing multi-dimensional topk indexes

        Returns:
            - residuals
        
        Example:
            - acc = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
            - indexes = (array([0, 2]), array([2, 1]))
            - output: [[1, 2, 0], [4, 5, 6], [7, 0, 9]]
        """

        if method == "numpy":
            residuals = np.array(acc)
            residuals[indexes] = 0
            return residuals


    
    def _ok_sparse_allreduce(self, acc, t, k, space_repartition_t=64, thresholds_re_evaluation_t=32):
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
            self.local_th = self._th_re_evaluate(acc, k)
        
        if t % space_repartition_t == 0:
            self.boundaries = self._space_repartition(acc, self.local_th)

        reduced_topk, local_topk_indexes = self._split_and_reduce(acc, self.local_th, self.boundaries)
        
        if t % thresholds_re_evaluation_t == 0:
            all_reduced_topk = self._allgather(reduced_topk)
            self.global_th = self._th_re_evaluate(all_reduced_topk, k)

        u, global_topk_indexes = self._balance_and_allgather(reduced_topk, self.global_th)
        indexes = self.intersect_indexes(local_topk_indexes, global_topk_indexes, acc.shape)
        return u, indexes


    def _th_re_evaluate(self, acc, k):
        """
        Return the absolute gradient threshold of acc matrix

        Parameters:
            - acc: gradient dense matrix
            - k: selection values hyperparameter

        Returns:
            - threshold: absolute gradient threshold k 
        """

        sorted_acc = np.sort(np.abs(acc).flatten())
        if len(sorted_acc) > k:
            threshold = sorted_acc[-k]
        else:
            # If k is larger than the total acc: select the smallest value
            threshold = sorted_acc[0]
        return threshold


    def _space_repartition(self, acc, local_th):
        """
        Returns the boundaries of the regions of the gradient matrix for the split and reduce phase.
        
        TODO: Currently, the distribution of space is not balanced, 
        but static: the matrix is divided into P equal regions.
        The regions should not be static, they should be distributed 
        in a balance way regarding to topk values locations.

        Parameters:
            - acc: gradient matrix values
            - local_th: local process gradient threshold

        Returns:
            - boundaries: { proc_id : (row_start, row_end) }
                where row_start is included and row_end is excluded.
        """

        boundaries = {}
        total_rows = acc.shape[0]
        region_size = total_rows // self.nprocs

        for proc in range(self.nprocs):
            row_start = proc * region_size
            row_end = row_start + region_size
            if proc == self.nprocs - 1:  
                row_end = total_rows
            boundaries[proc] = (row_start, row_end)
        return boundaries


    def _split_and_reduce(self, acc, local_th, boundaries):
        """
        Split the gradients into partitions and reduce them by selecting top-k values.
        Each worker receives sparse regions from the other workers and and then conducts
        the reduction locally. 

        TODO: To simplify the implementation, boundaries are not used, instead: allreduce
        TODO: To simplify the implementation, destination rotation and bucketing,
        is not yet implemented.

        Parameters:

            - acc: Gradient matrix accumulation values.
            - local_th: Local threshold for selecting top-k values.
            - boundaries: Boundaries for partitioning the gradient space: { proc_id : (row_start, row_end) }.

         Returns:
            - reduced_topk: The reduced top-k gradient values.
            - local_topk_indexes: The indices of the top-k gradient values selected locally.
        """
        # 1. Local topk values selection: 
        # All procs perform topk selection in all acc
        local_topk, local_topk_indexes = self._top_threshold_selection(acc, local_th)

        # 2. Balance split and reduce
        # TODO: Para simplificar de momento: Allreduce
        reduced_topk = self._allreduce(local_topk)
        return reduced_topk, local_topk_indexes


    def _balance_and_allgather(self, reduced_topk, global_th):
        # 1. Global topk selection
        global_topk, global_topk_indexes = self._top_threshold_selection(reduced_topk, global_th)

        # 2. Data packaging
        # TODO

        # 3. Data balancing
        # TODO

        # 4. Allgatherv using recursive doubling
        # Como antes hemos hecho un allreduce, creo que no hace falta allgather
        # TODO 

        return global_topk, global_topk_indexes


    def intersect_indexes(self, local_indexes_tuple, global_indexes_tuple, shape, method="numpy"):
        """
        Calculates the intersection of two sets of indices of any dimension.

        Parameters:
            - local_indexes_tuple: a tuple of N arrays of shape M. 
            - global_indexes_tuple: a tuple of N arrays of shape M. 
                where N is the dimensions and M the number of indexes.
        
        Returns:
            - Set of tuples representing the common indices in all dimensions.
        
        Example:
            - local_indexes_tuple = (np.array([0, 3, 2]), np.array([2, 4, 1]), np.array([5, 1, 5]))
            - global_indexes_tuple = (np.array([1, 2, 3]), np.array([3, 1, 4]), np.array([5, 6, 1]))
            - output: (array([3]), array([4]), array([1]))
        """
        
        if method == "numpy":
            local_flattened_indexes = np.ravel_multi_index(local_indexes_tuple, dims=shape)
            global_flattened_indexes = np.ravel_multi_index(global_indexes_tuple, dims=shape)
            flattened_intersection = np.intersect1d(local_flattened_indexes, global_flattened_indexes)
            unravel_intersection = np.unravel_index(flattened_intersection, shape=shape)
            return unravel_intersection
            

    # def _topk(self, tensor, k):
    #     """
    #         Implementation of Li and Hoefler
    #     """
    #     indexes = np.abs(tensor).argsort()[-k:]
    #     return indexes, tensor[indexes]


    def _top_threshold_selection(self, data, threshold, method="numpy"):
        """
        Selects top-k elements from the data that are greater than or equal to the threshold.
        
        Parameters:
            - data: The input tensor from which to select elements.
            - threshold (float): The threshold value to compare against the absolute values of the data elements.
        
        Returns:
            - topk: A tensor of the same shape as `data`, where elements that meet the threshold condition
                        are retained, and all other elements are set to zero.
            - topk_indexes: a tuple of np.arrays with the indexes, e.g (array([0, 1]), array([1, 1]))
        """

        topk, topk_indexes = None, None

        if method == "numpy":
            topk = np.zeros_like(data, dtype=data.dtype)
            topk_indexes = np.where(np.abs(data) >= threshold)
            topk[topk_indexes] = data[topk_indexes]

        elif method == "cython":
            topk, topk_indexes = top_threshold_selection_cython(data, threshold)

        elif method == "cython_flattening":
            topk, topk_indexes = flattened_top_threshold_selection_cython(data.flatten(), threshold)
            topk = np.reshape(topk, data.shape)
            topk_indexes = np.unravel_index(topk_indexes, data.shape)

        return topk, topk_indexes
        

    def _allgather(self, data, method="sparse"):
        if self.nprocs == 1:
            return data
        
        elif method == "dense":
            # TODO: ¿Por qué funciona el siguiente allgather? 
            # No debería dar error MPI.IN_PLACE, ya que re comunica data.size * self.nprocs
            self.comm.Allgather(MPI.IN_PLACE, data)
            return data
        
        elif method == "sparse":
            # TODO: Funciona pero converge raro
            indexes, values = self._convert_to_coo_format(data)
            self.comm.Allgather(MPI.IN_PLACE, indexes)
            self.comm.Allgather(MPI.IN_PLACE, values)
            dense_data = self._convert_from_coo_format([indexes, values], data.shape)
            return dense_data


    def _allreduce(self, data, method="dense"):
        if self.nprocs == 1:
            return data

        elif method == "dense":
            self.comm.Allreduce(MPI.IN_PLACE, data, op=MPI.SUM)
            return data
        
        elif method == "sparse":
            indexes, values = self._convert_to_coo_format(data)
            # TODO: Own allreduce implmentation
            dense_data = self._convert_from_coo_format([indexes, values], data.shape)
            return dense_data



    def _convert_to_coo_format(self, data, storage_format="offset"):
        """
        Returns a sparse array in coo format for tensors of any dimension

        Parameters: 
            - data: dense tensor of any dimensions
            - storage_format: "offset" (default) or "coordinate" 
        
        Returns:
            - if "offset" storage format: [indexes, non-zero values]
            - if "coordinate" storage format: [dim1 indexes, dim2 indexes, ..., non-zero values]
        """

        coo_storage_formats = {"offset", "coordinate"}

        if storage_format not in coo_storage_formats:
            print(f"Storage format '{storage_format}' not known. Try with: {coo_storage_formats}")
            return None

        if storage_format == "coordinate":
            indices = np.nonzero(data)
            values = data[indices]
            coordinates = [indices[dim] for dim in range(data.ndim)]
            coordinates.append(values)
            coo_data = coordinates

        elif storage_format == "offset":
            flattened_data = data.flatten()
            indexes = np.nonzero(flattened_data)[0]
            values = flattened_data[indexes]
            coo_data = [indexes, values]

        return coo_data


    def _convert_from_coo_format(self, coo_data, shape, dtype=np.float32, storage_format="offset"):
        """
        Returns a dense array from sparse data 

        Parameters: 
            - coo_data: sparse matrix data in format [[index], [value]] for "offset" 
                        or [[row], [col], ..., [value]] for "coordinate"
            - shape: dense matrix expected shape 
            - dtype: data type of the output array, default is np.float32
            - storage_format: "offset" (default) or "coordinate" 
        
        Returns:
            - dense_matrix: dense data with the provided shape
        """

        coo_storage_formats = {"offset", "coordinate"} 

        if storage_format not in coo_storage_formats:
            print(f"Storage format '{storage_format}' not known. Try with: {coo_storage_formats}")
            return None

        dense_matrix = np.zeros(shape, dtype=dtype)

        if storage_format == "coordinate":
            coordinates = coo_data[:-1]  
            values = coo_data[-1]  
            for idx, value in zip(zip(*coordinates), values):
                dense_matrix[idx] = value

        elif storage_format == "offset":
            indices, values = coo_data
            total_elements = np.prod(shape)  
            for offset, value in zip(indices, values):
                index = np.unravel_index(offset, shape)
                dense_matrix[index] = value

        return dense_matrix
