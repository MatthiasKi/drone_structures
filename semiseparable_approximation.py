import numpy as np
from tvsclib.mixed_system import MixedSystem
from tvsclib.toeplitz_operator import ToeplitzOperator
from tvsclib.system_identification_svd import SystemIdentificationSVD
from phoenix_drone_simulation.utils.utils import load_network_json

class SSSApproximator:
    def get_nb_parameters(self, state_space_dim: int, dims_in: list, dims_out: list):
        params = 0
        for (dim_in, dim_out) in zip(dims_in, dims_out):
            params += 2*np.power(state_space_dim, 2) + dim_in * dim_out + 2 * state_space_dim * dim_in + 2 * state_space_dim * dim_out
        return params

    def get_max_dim_according_to_max_nb_parameters(self, max_nb_parameters: int, dims_in: list, dims_out: list):
        curr_state_space_dim = 1
        while True:
            params = self.get_nb_parameters(state_space_dim=curr_state_space_dim, dims_in=dims_in, dims_out=dims_out)
            if params < max_nb_parameters:
                curr_state_space_dim += 1
            else:
                return curr_state_space_dim

    def semiseparable_approximation(self, weights: np.ndarray, max_nb_free_parameters: int):
        assert len(weights.shape) == 2, "Weights must describe a matrix (not a tensor of arbitrary shape)"

        dim_lengths = min(weights.shape)

        dims_in = np.ones((dim_lengths,), dtype='int32')
        dims_out = np.ones((dim_lengths,), dtype='int32')

        nb_two_dim_inputs = 0
        nb_two_dim_outputs = 0

        # Fix dims for non-square matrices
        if weights.shape[1] > weights.shape[0]:
            assert weights.shape[1] - weights.shape[0] < weights.shape[0], "Can not handle weight matrices with more than twice outputs than inputs"
            nb_two_dim_inputs = weights.shape[1]-weights.shape[0]
            dims_in[:nb_two_dim_inputs] = 2
        elif weights.shape[0] > weights.shape[1]:
            assert weights.shape[0] - weights.shape[1] < weights.shape[1], "Can not handle weight matrices with more than twice inputs than outputs"
            nb_two_dim_outputs = weights.shape[0]-weights.shape[1]
            dims_out[:nb_two_dim_outputs] = 2

        assert np.sum(dims_in) == weights.shape[1], "Not enough input dimensions"
        assert np.sum(dims_out) == weights.shape[0], "Not enough output dimensions"
        assert nb_two_dim_inputs == 0 or nb_two_dim_outputs == 0, "It's currently only supported that either inputs or outputs are two-dimensional"

        max_dim = self.get_max_dim_according_to_max_nb_parameters(max_nb_free_parameters, dims_in=dims_in, dims_out=dims_out)
        print("Max State-Space Dim: " + str(max_dim))
        print("State-Space Number free parameters: " + str(self.get_nb_parameters(state_space_dim=1, dims_in=dims_in, dims_out=dims_out)))

        T = ToeplitzOperator(weights, dims_in, dims_out)
        S = SystemIdentificationSVD(toeplitz=T, max_states_local=max_dim)
        system_approx = MixedSystem(S)

        return system_approx.to_matrix()

    def calc_input_output_dims(self, weights: np.ndarray):
        assert len(weights.shape) == 2, "Weights must describe a matrix (not a tensor of arbitrary shape)"

        dim_lengths = min(weights.shape)

        dims_in = np.ones((dim_lengths,), dtype='int32')
        dims_out = np.ones((dim_lengths,), dtype='int32')

        nb_two_dim_inputs = 0
        nb_two_dim_outputs = 0

        # Fix dims for non-square matrices
        if weights.shape[1] > weights.shape[0]:
            base_inp_dim = int(weights.shape[1] / weights.shape[0])
            dims_in *= base_inp_dim
            nb_larger_dims = weights.shape[1] - weights.shape[0] * base_inp_dim
            dims_in[:nb_larger_dims] = base_inp_dim + 1
        elif weights.shape[0] > weights.shape[1]:
            base_out_dim = int(weights.shape[0] / weights.shape[1])
            dims_out *= base_out_dim
            nb_larger_dims = weights.shape[0] - weights.shape[1] * base_out_dim
            dims_out[:nb_larger_dims] = base_out_dim + 1

        assert np.sum(dims_in) == weights.shape[1], "Not enough input dimensions"
        assert np.sum(dims_out) == weights.shape[0], "Not enough output dimensions"
        assert nb_two_dim_inputs == 0 or nb_two_dim_outputs == 0, "It's currently only supported that either inputs or outputs are two-dimensional"

        return dims_in, dims_out, nb_two_dim_inputs, nb_two_dim_outputs

    def approximate_simple(self, weights: np.ndarray):
        dim_lengths = min(weights.shape)
        dims_in, dims_out, nb_two_dim_inputs, nb_two_dim_outputs = self.calc_input_output_dims(weights)

        T = ToeplitzOperator(weights, dims_in, dims_out)
        S = SystemIdentificationSVD(toeplitz=T, max_states_local=1)
        system_approx = MixedSystem(S)

        res_dict = dict()
        res_dict["type"] = "SemiseparableApproximator"
        res_dict["approx_mat_dense"] = system_approx.to_matrix()
        res_dict["nb_parameters"] = self.get_nb_parameters(state_space_dim=1, dims_in=dims_in, dims_out=dims_out)
        return res_dict

    def benchmark_approximate(self, target_mat: np.ndarray, param_share: float, log_filepath: str):
        dims_in, dims_out, nb_two_dim_inputs, nb_two_dim_outputs = self.calc_input_output_dims(target_mat)

        max_nb_free_parameters = int(target_mat.size * param_share)
        state_dim = self.get_max_dim_according_to_max_nb_parameters(max_nb_free_parameters, dims_in=dims_in, dims_out=dims_out)

        T = ToeplitzOperator(target_mat, dims_in, dims_out)
        S = SystemIdentificationSVD(toeplitz=T, max_states_local=state_dim)
        system_approx = MixedSystem(S)

        res_dict = dict()
        res_dict["type"] = "SemiseparableApproximator"
        res_dict["approx_mat_dense"] = system_approx.to_matrix()
        res_dict["nb_parameters"] = self.get_nb_parameters(state_space_dim=state_dim, dims_in=dims_in, dims_out=dims_out)
        return res_dict

if __name__ == "__main__":
    # Test case
    model_filename = "data/12hiddenneurons/model.json"
    net = load_network_json(model_filename)

    target_mat = net[2].weight.detach().numpy().astype("float32")

    approximator = SSSApproximator()
    res_dict = approximator.benchmark_approximate(target_mat=target_mat, param_share=0.8, log_filepath="tmp.log")
    norm_difference = np.linalg.norm(target_mat - res_dict["approx_mat_dense"], ord="fro")
    print(norm_difference)
    print("Norm of the target matrix: " + str(np.linalg.norm(target_mat, ord="fro")))

    halt = 1