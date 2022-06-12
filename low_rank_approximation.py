import numpy as np
from sklearn.utils.extmath import randomized_svd
from phoenix_drone_simulation.utils.utils import load_network_json

class LowRankApproximator:
    def get_max_nb_singular_values_wrt_max_nb_operations(self, weights_shape: np.ndarray, max_nb_operations: int):
        return int((max_nb_operations + weights_shape[0]) / (2 * weights_shape[1] - 1 + 2 * weights_shape[0]))

    def get_max_nb_singular_values_wrt_max_nb_free_parameters(self, weights_shape: np.ndarray, max_nb_free_parameters: int):
        return int(max_nb_free_parameters / (weights_shape[0] + weights_shape[1]))

    def get_nb_operations(self, weights_shape: np.ndarray, nb_singular_values: int):
        return 2 * weights_shape[1] * nb_singular_values - nb_singular_values + 2 * weights_shape[0] * nb_singular_values - weights_shape[0]

    def low_rank_approximation(self, weights: np.ndarray, nb_singular_values: int):
        U, s, Vh = randomized_svd(M=weights, n_components=nb_singular_values)

        # NOTE: It seems to make a difference if you use diag or only *: (U*s)@Vh works, but (U*np.sqrt(s))@(np.sqrt(s)*Vh) nor U@(s*Vh) seem to work
        left_side = U @ np.diag(np.sqrt(s))
        right_side = np.diag(np.sqrt(s)) @ Vh

        return (left_side, right_side), left_side @ right_side

    def approximate_simple(self, weights: np.ndarray):
        lr_approximation, dense_approximation = self.low_rank_approximation(weights, 1)
        res_dict = dict()
        res_dict["type"] = "LowRankApproximator"
        res_dict["lr_mat"] = lr_approximation
        res_dict["approx_mat_dense"] = dense_approximation
        res_dict["nb_parameters"] = self.get_nb_operations(weights.shape, 1)
        return res_dict

    def benchmark_approximate(self, target_mat: np.ndarray, param_share: float, log_filepath: str):
        nb_singular_values = self.get_max_nb_singular_values_wrt_max_nb_free_parameters(target_mat.shape, int(target_mat.size * param_share))

        lr_approximation, dense_approximation = self.low_rank_approximation(weights=target_mat, nb_singular_values=nb_singular_values)
        res_dict = dict()
        res_dict["type"] = "LowRankApproximator"
        res_dict["lr_mat"] = lr_approximation
        res_dict["approx_mat_dense"] = dense_approximation
        res_dict["nb_parameters"] = self.get_nb_operations(target_mat.shape, nb_singular_values)
        return res_dict

if __name__ == "__main__":
    # Test case
    model_filename = "data/30hiddenneurons/model.json"
    net = load_network_json(model_filename)

    target_mat = net[2].weight.detach().numpy().astype("float64")

    approximator = LowRankApproximator()
    res_dict = approximator.benchmark_approximate(target_mat=target_mat, param_share=0.8, log_filepath="tmp.log")
    norm_difference = np.linalg.norm(target_mat - res_dict["approx_mat_dense"], ord="fro")
    print(norm_difference)
    print("Norm of the target matrix: " + str(np.linalg.norm(target_mat, ord="fro")))
    
    # Test for the low rank decomposition
    if target_mat.shape[0] == target_mat.shape[1]:
        lr_approximation, dense_mat = approximator.low_rank_approximation(target_mat, target_mat.shape[0])
        assert np.allclose(target_mat, dense_mat), "Even with all singular values, the low rank approximation is not close to the target matrix"
    
    halt = 1