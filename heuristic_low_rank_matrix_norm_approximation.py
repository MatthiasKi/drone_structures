import numpy as np
from phoenix_drone_simulation.utils.utils import load_network_json
import torch

# This is done here similar to the TD-LDR approach (but using adam, thats why i can skip the different alpha-iteations)
class LowRankMatrixNormApproximator:
    def __init__(self, norm):
        self.possible_special_norms = ["nuc", "fro"]
        assert isinstance(norm, int) or isinstance(norm, float) or norm == np.inf or norm == -np.inf or (isinstance(norm, str) and norm in self.possible_special_norms), "Asked norm is not implemented: " + str(norm)

        self.norm = norm

    def get_max_nb_singular_values_wrt_max_nb_operations(self, weights_shape: np.ndarray, max_nb_operations: int):
        return int((max_nb_operations + weights_shape[0]) / (2 * weights_shape[1] - 1 + 2 * weights_shape[0]))

    def get_max_nb_singular_values_wrt_max_nb_free_parameters(self, weights_shape: np.ndarray, max_nb_free_parameters: int):
        return int(max_nb_free_parameters / (weights_shape[0] + weights_shape[1]))

    def get_nb_operations(self, weights_shape: np.ndarray, nb_singular_values: int):
        return 2 * weights_shape[1] * nb_singular_values - nb_singular_values + 2 * weights_shape[0] * nb_singular_values - weights_shape[0]

    def get_glorot_std(self, shape: tuple):
        if len(shape) == 2:
            glorot_std = np.sqrt(6.0 / (shape[0] + shape[1]))
        elif len(shape) == 1:
            glorot_std = np.sqrt(6.0 / shape[0])
        else:
            raise Exception("Shape length mismatch: " + str(len(shape)))
        return glorot_std

    def get_init_random_matrix_torch(self, shape: tuple):
        glorot_std = self.get_glorot_std(shape)
        curr_mat = torch.tensor(np.random.uniform(-glorot_std, glorot_std, size=shape))
        curr_mat.requires_grad_()
        return curr_mat

    def approx_loss_torch(self, matrices: list, target_mat: np.ndarray):
        curr_mat = torch.matmul(matrices[0], matrices[1])
        return torch.norm(torch.tensor(target_mat) - curr_mat, p=self.norm)

    def approximate_wrt_matrix_norm(self, target_mat: np.ndarray, rank: int, min_optimization_epsilon=1e-4, verbose=False):
        if verbose:
            print("Starting to approximate target matri wrt " + str(self.norm) + " norm and rank " + str(rank))

        matrices = [
            self.get_init_random_matrix_torch((target_mat.shape[0], rank)),
            self.get_init_random_matrix_torch((rank, target_mat.shape[1]))
        ]

        optimizer = torch.optim.Adam(matrices, lr=1e-1)
        last_loss = 1e9
        loss = 1e8
        optim_step = 1
        while last_loss - loss > min_optimization_epsilon:
            last_loss = loss
            optimizer.zero_grad()
            loss = self.approx_loss_torch(matrices=matrices, target_mat=target_mat)
            if verbose:
                print('Optimization Step # {}, loss: {}'.format(optim_step, loss.item()))
            loss.backward()
            
            optimizer.step()
            optim_step += 1

        return matrices

    def benchmark_approximate(self, target_mat: np.ndarray, param_share: float, log_filepath: str):
        nb_singular_values = self.get_max_nb_singular_values_wrt_max_nb_free_parameters(target_mat.shape, int(target_mat.size * param_share))
        
        best_approximation_lr_mats = None
        best_approximation_approx_mat_dense = None
        best_loss = None
        for _ in range(5):
            lr_mats = self.approximate_wrt_matrix_norm(target_mat=target_mat, rank=nb_singular_values, verbose=False)
            curr_loss = self.approx_loss_torch(matrices=lr_mats, target_mat=target_mat)
            if best_loss is None or curr_loss < best_loss:
                best_loss = curr_loss
                best_approximation_lr_mats = [mat.detach().numpy() for mat in lr_mats]
                best_approximation_approx_mat_dense = best_approximation_lr_mats[0] @ best_approximation_lr_mats[1]

        res_dict = dict()
        res_dict["type"] = "LowRankMatrixNormApproximator"
        res_dict["lr_mats"] = best_approximation_lr_mats
        res_dict["approx_mat_dense"] = best_approximation_approx_mat_dense
        res_dict["nb_parameters"] = self.get_nb_operations(target_mat.shape, nb_singular_values)
        return res_dict

if __name__ == "__main__":
    # Test case
    model_filename = "data/30hiddenneurons/model.json"
    net = load_network_json(model_filename)

    target_mat = net[2].weight.detach().numpy().astype("float64")

    test_norms = [-1, 1, -2, 2, np.inf, -np.inf, "nuc", "fro"]

    for norm in test_norms:
        approximator = LowRankMatrixNormApproximator(norm=norm)
        res_dict = approximator.benchmark_approximate(target_mat=target_mat, param_share=0.5, log_filepath="tmp.log")
        norm_difference = np.linalg.norm(target_mat - res_dict["approx_mat_dense"], ord=norm)
        print("--- Norm Checked: " + str(norm) + " ---")
        print("Norm Difference: " + str(norm_difference) + " (Norm of the target matrix: " + str(np.linalg.norm(target_mat, ord=norm)) + ")")
        
    halt = 1