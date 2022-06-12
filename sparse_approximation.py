import numpy as np
from scipy.sparse import coo_matrix

class SparseApproximator:        
    def get_max_nb_elements_wrt_max_nb_operations(self, max_nb_operations: int):
        return int(max_nb_operations / 2)

    def get_max_nb_elements_wrt_max_nb_free_parameters(self, max_nb_free_parameters: int):
        return max_nb_free_parameters

    def get_nb_operations(self, nb_nonzero_elements: int):
        return 2 * nb_nonzero_elements

    def sparse_approximation(self, weights: np.ndarray, nb_nonzero_elements: int):
        # Get the nb_nonzero_element biggest values
        biggest_elements_pos = np.argsort(np.abs(weights.flatten()) * -1)
        
        # Get the indices of these positions
        biggest_elements_idx = np.unravel_index(biggest_elements_pos, weights.shape)

        # Build the approximation mat
        approx_mat = coo_matrix((weights.flatten()[biggest_elements_pos[:nb_nonzero_elements]], (biggest_elements_idx[0][:nb_nonzero_elements], biggest_elements_idx[1][:nb_nonzero_elements])), shape=weights.shape)

        # Return the dense approximated mat
        return approx_mat, approx_mat.todense()

    def approximate_simple(self, weights: np.ndarray):
        n = min(weights.shape)
        res_dict = dict()
        sparse_mat, approx_mat_dense = self.sparse_approximation(weights, n)
        res_dict["type"] = "SparseApproximator"
        res_dict["sparse_mat"] = sparse_mat
        res_dict["approx_mat_dense"] = approx_mat_dense
        res_dict["nb_parameters"] = n
        return res_dict

    def benchmark_approximate(self, target_mat: np.ndarray, param_share: float, log_filepath: str):
        nb_nonzero_elements = int(target_mat.size * param_share)
        sparse_mat, approx_mat_dense = self.sparse_approximation(weights=target_mat, nb_nonzero_elements=nb_nonzero_elements)
        
        res_dict = dict()
        res_dict["type"] = "SparseApproximator"
        res_dict["sparse_mat"] = sparse_mat
        res_dict["approx_mat_dense"] = approx_mat_dense
        res_dict["nb_parameters"] = nb_nonzero_elements
        return res_dict

if __name__ == "__main__":
    nb_nonzero_elements = 500

    # Test case
    random_matrix = np.random.uniform(low=-1,high=1,size=(100,50))
    approximator = SparseApproximator()
    sparse_mat, approx_mat_dense = approximator.sparse_approximation(random_matrix, nb_nonzero_elements)
    norm_difference = np.linalg.norm(random_matrix - approx_mat_dense, ord="fro")
    print(norm_difference)

    assert len(sparse_mat.data) == nb_nonzero_elements

    halt = 1