import warnings
import numpy as np

from pydtnn.backends.cpu.utils.sparse_cython import \
    top_threshold_selection_dense_cython, \
    top_threshold_selection_coo_cython, \
    summ_coo_cython


class SparseMatrixCOO:
    """
    Represents a sparse matrix in COO format.
    This format stores the matrix using three arrays:
        - data: the nonzero values.
        - row: the row indices corresponding to each value.
        - col: the column indices corresponding to each value.
    The matrix is assumed to be in canonical format: indices sorted by row and then by column, and no duplicate entries are present.
    This class is not designed to store explict zeros so, len(self.data) should always be equal to nnz. 
    """

    def __init__(self, data, row, col, shape, has_canonical_format):
        """
        Primary initializer for SparseMatrixCOO.

        Parameters:
            data (np.ndarray): Array with the nonzero values.
            row  (np.ndarray): Array with the row indices.
            col  (np.ndarray): Array with the column indices.
            shape (tuple): Shape of the original matrix.
        """

        if len(data) != len(row) or len(data) != len(col):
            raise AssertionError("Data, row, and col arrays must have the same shape")

        if has_canonical_format:
            self.data = data
            self.row = row
            self.col = col
            self.shape = shape
            self.nnz = len(self.data)
            self.has_canonical_format = True
            assert (self._has_canonical_format())

        else:
            # TODO: order arrays in canonical format
            raise NotImplementedError("Not yet implemented constructor with unordered rows and cols")

    @classmethod
    def from_dense(cls, dense_array):
        """
        Alternative constructor to create a SparseMatrixCOO from a dense array.
        Only stores non-zero values!

        Parameters:
            dense_array (np.ndarray): A 2D dense matrix.

        Returns:
            SparseMatrixCOO: The sparse matrix in COO format
        """

        if len(dense_array.shape) != 2:
            raise AssertionError("Dense array must be 2D.")

        warnings.warn("From dense constructor should be used only in case of debugging for performance reasons.")

        row, col = np.where(dense_array != 0)
        data = dense_array[row, col]
        row = row.astype(np.int32)
        col = col.astype(np.int32)
        return cls(data, row, col, dense_array.shape, has_canonical_format=True)

    @classmethod
    def from_dense_top_selection(cls, dense_array, threshold):
        """
        Alternative constructor to create a SparseMatrixCOO from a dense array,
        considering only elements with absolute value greater than or equal to the threshold.

        Parameters:
            dense_array (np.ndarray): A 2D dense matrix.
            threshold (float): Threshold for including an element.

        Returns:
            SparseMatrixCOO: The sparse matrix in COO format, containing only significant elements.
        """

        if len(dense_array.shape) != 2:
            raise AssertionError("Dense array must be 2D.")

        # topk_row, topk_col = np.where(np.abs(dense_array) >= threshold)
        # topk = dense_array[topk_row, topk_col]
        topk, topk_row, topk_col = top_threshold_selection_dense_cython(dense_array, threshold)
        return cls(topk, topk_row, topk_col, dense_array.shape, has_canonical_format=True)

    def top_selection(self, threshold, inplace=True):
        """
        Performs top threshold selection on sparse array

        Parameters:
            threshold (float): Threshold for including an element.
            inplace (bool, optional):

        Returns:
            topk (SparseMatrixCOO): if inplace == False, or void (None): if inplace == True 
        """

        topk, topk_row, topk_col = top_threshold_selection_coo_cython(self.data, self.row, self.col, threshold)

        if inplace:
            self.data = topk
            self.row = topk_row
            self.col = topk_col
            self.nnz = len(self.data)
            # self.shape remains equal
            # self.has_canonical_format remains equal
        else:
            return SparseMatrixCOO(topk, topk_row, topk_col, self.shape, self.has_canonical_format)

    def get_indexes(self):
        """
        Returns the row and col

        Returns:
            tuple (tuple): row and col
        """
        return self.row, self.col

    def get_triplet(self):
        """
        Returns the data, row, col triplet

        Returns:
            triplet (tuple): data, row col
        """
        return self.data, self.row, self.col

    def slice(self, row_start, row_end, reset_indexes=False):
        """
        Perform a slice by row of the sparse matrix.

        Parameters:
            row_start (int): The starting row index (inclusive) of the slice.
            row_end (int): The ending row index (exclusive) of the slice.
            reset_indexes (bool, optional): If True, resets the row indices of the 
                                            sliced matrix so that `row_start` maps to zero.
                                            Defaults to False.
        Returns:
            coo_sliced: (SparseMatrixCOO): A row-sliced sparse matrix of self
        """
        start_index = np.searchsorted(self.row, row_start, side='left')
        ending_index = np.searchsorted(self.row, row_end, side='left')

        sliced_data = self.data[start_index:ending_index]
        sliced_row = self.row[start_index:ending_index]
        sliced_col = self.col[start_index:ending_index]
        if reset_indexes:
            sliced_row -= row_start

        return SparseMatrixCOO(sliced_data, sliced_row, sliced_col, self.shape, self.has_canonical_format)

    def to_dense(self):
        """
        Convert to dense np.array

        Returns:
            dense_matrix: (np.array): a dense matrix
        """

        warnings.warn("This function ('to_sparse') should be used only in case of debugging for performance reasons.")

        dense_matrix = np.zeros(self.shape, dtype=np.float32)
        dense_matrix[self.row, self.col] = self.data
        return dense_matrix

    def __add__(self, other):
        """
        Adds two SparseMatrixCOO matrices that are in canonical format.

        Parameters:
            other (SparseMatrixCOO): Another SparseMatrixCOO instance.

        Returns:
            SparseMatrixCOO: A new instance representing the sum of both matrices.
        """

        if type(other) != SparseMatrixCOO:
            raise AssertionError("Operand must be a SparseMatrixCOO instance.")
        if self.shape != other.shape:
            raise AssertionError("Matrices must have the same shape.")
        if not self.has_canonical_format or not other.has_canonical_format:
            raise AssertionError("Both matrices must be in canonical format.")

        summ_val, summ_row, summ_col = summ_coo_cython(self.data, self.row, self.col, other.data, other.row, other.col)
        return SparseMatrixCOO(summ_val, summ_row, summ_col, self.shape, has_canonical_format=True)

    def __radd__(self, other):
        """
        Implements right-hand addition to support the built-in sum() function.

        This method allows an instance of this class to be used with sum() by handling the
        case where the left operand is 0. If 'other' is 0, it returns the instance itself;
        otherwise, it delegates the operation to the __add__ method.

        Parameters:
            other (int or instance of the same class): The left-hand operand, typically 0 when used with sum().

        Returns:
            An instance of the class representing the sum of self and other.
        """
        if other == 0:
            return self
        return self.__add__(other)

    def _has_canonical_format(self):
        """
        Check if SparseMatrixCOO follows canonical format: 
            - Indexes are sorted by row and then by column 
            - There are no duplicate entries
            - There may have explicit zero elements
        This function is computationally expensive and therefore should only be used for developing/debugging purposes.
        This function should only be used in developement to assert that sparse matrices have canonical format. 

        Returns:
            has_canonical_format (boolean): True if indexes are in canonical format, False if not. 
        """

        warnings.warn("This function ('has_canonical_format') should be used only in case of debugging for performance reasons.")

        if self.nnz == 0:
            return True

        if not np.all(self.row[:-1] <= self.row[1:]):
            return False

        for i in range(self.nnz - 1):
            if self.row[i] == self.row[i + 1] and self.col[i] >= self.col[i + 1]:
                return False
        return True
