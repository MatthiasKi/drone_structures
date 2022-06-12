import numpy as np
import math
from phoenix_drone_simulation.utils.utils import load_network_json
from pyfaust.fact import hierarchical
from pyfaust.proj import sp
from pyfaust.factparams import ParamsHierarchical, StoppingCriterion
from scipy.optimize import minimize_scalar

class FaustApproximator:
    def __init__(self, nb_matrices: int, linear_nb_nonzero_elements_distribution=True):
        assert nb_matrices > 1, "Nb of matrices must be greater than one"
        self.nb_matrices = nb_matrices
        self.linear_nb_nonzero_elements_distribution = linear_nb_nonzero_elements_distribution

    def get_nb_elements_list(self) -> list:
        if self.nb_matrices == 2:
            nb_nonzero_elements_list = [self.total_nb_nonzero_elements-self.last_nb_nonzero_elements, self.last_nb_nonzero_elements]
        else:
            # NOTE if linear is false, then the residual shaping is done exponentially
            k = self.nb_matrices - 1
            if self.linear_nb_nonzero_elements_distribution:
                alpha = (2 * self.total_nb_nonzero_elements) / (k+1) - self.last_nb_nonzero_elements
                m = (self.last_nb_nonzero_elements - alpha) / k
                nb_nonzero_elements_list = [int(alpha + m*mat_i) for mat_i in range(self.nb_matrices)]
            else:
                optim_res = minimize_scalar(self.exponential_distribution_optimization_fun, bounds=(1, self.total_nb_nonzero_elements), method='bounded')
                alpha = optim_res.x
                m = self.alpha_to_m(alpha)
                nb_nonzero_elements_list = [int(alpha * np.exp(m * mat_i)) for mat_i in range(self.nb_matrices)]
        
        # Due to rounding errors, there might be minimal errors in the number elements list
        nb_exceeded_elements = sum(nb_nonzero_elements_list) - self.total_nb_nonzero_elements
        while nb_exceeded_elements > 0:
            nb_elements_pos_to_alter = nb_nonzero_elements_list.index(max(nb_nonzero_elements_list))
            nb_nonzero_elements_list[nb_elements_pos_to_alter] -= nb_exceeded_elements
            nb_exceeded_elements = sum(nb_nonzero_elements_list) - self.total_nb_nonzero_elements

        assert len(nb_nonzero_elements_list) == self.nb_matrices, "Length of the nb_nonzero_elements_list does not match the expected length"
        assert nb_nonzero_elements_list[-1] <= self.last_nb_nonzero_elements, "The last entry in the nb_nonzero_elements_list does not match the expected last nb of nonzero elements"
        assert sum(nb_nonzero_elements_list) <= self.total_nb_nonzero_elements, "The sum over the element distribution deviates from the expected total number of elements"

        if any([tmp < 0 for tmp in nb_nonzero_elements_list]):
            nb_nonzero_elements_list = [0 for _ in range(self.nb_matrices)]

        return nb_nonzero_elements_list

    def alpha_to_m(self, alpha):
        k = self.nb_matrices - 1
        arg = self.last_nb_nonzero_elements / alpha
        if arg > 1e-6:
            return np.log(arg) / k
        else:
            return -14 / k

    def exponential_distribution_optimization_fun(self, x):
        nb_nonzero_elements_integral = 0
        for mat_i in range(self.nb_matrices):
            nb_nonzero_elements_integral += x * np.exp(self.alpha_to_m(x) * mat_i)
        return np.abs(nb_nonzero_elements_integral - self.total_nb_nonzero_elements)

    def get_matrices_shapes_list(self, target_mat_shape: np.ndarray):
        max_shape_dim = max(target_mat_shape)
        matrices_shapes_list = [(target_mat_shape[0], max_shape_dim)] \
            + [(max_shape_dim, max_shape_dim) for _ in range(self.nb_matrices - 2)] \
            + [(max_shape_dim, target_mat_shape[1])]
        return matrices_shapes_list

    def benchmark_approximate(self, target_mat: np.ndarray, param_share: float, log_filepath: str):
        nb_nonzero_elements = int(target_mat.size * param_share)
        best_approximation = None
        for last_mat_param_share in np.linspace(0.1, 0.9, num=17):
            last_nb_nonzero_elements = int(last_mat_param_share * nb_nonzero_elements)
            res_dict = self.faust_approximation(weights=target_mat, last_nb_nonzero_elements=last_nb_nonzero_elements, total_nb_nonzero_elements=nb_nonzero_elements)
            
            if not (res_dict is None):
                norm = np.linalg.norm(target_mat - res_dict["approx_mat_dense"], ord="fro")

                if best_approximation is None or best_approximation["objective_function_result"] > norm:
                    res_dict["objective_function_result"] = norm
                    best_approximation = res_dict
        
        return best_approximation

    def faust_factorization_to_dense(self, faust_matrices: list, nb_nonzero_elements_list=None): 
        # The nb_nonzero_elements_list is only used to check if the number nonzero elements are right
        res = faust_matrices[0]
        assert nb_nonzero_elements_list is None or len(res.data) <= nb_nonzero_elements_list[0], "Number of nonzero elements is wrong"
        for mat_i, mat in enumerate(faust_matrices[1:]):
            assert nb_nonzero_elements_list is None or len(mat.data) <= nb_nonzero_elements_list[mat_i+1], "Number of nonzero elements is wrong"
            res = res @ mat
        if not isinstance(res, np.ndarray):
            res = res.todense()
        return res

    def faust_approximation(self, weights: np.ndarray, last_nb_nonzero_elements: int, total_nb_nonzero_elements: int):
        self.total_nb_nonzero_elements = total_nb_nonzero_elements
        self.last_nb_nonzero_elements = last_nb_nonzero_elements

        nb_elements_distribution = self.get_nb_elements_list()
        matrices_shapes_list = self.get_matrices_shapes_list(target_mat_shape=weights.shape)

        try:
            # Set constrains
            # The constrains define the constrains for non zero elements per factor
            # Details : https://faustgrp.gitlabpages.inria.fr/faust/last-doc/html/constraint.png
            # https://faust.inria.fr/api-doc/
            # NOTE That the last residual is then taken as the last matrix entry of the matrices_shapes_list
            fact_cons = []
            res_cons = []
            for factor_i, (factor_shape, factor_nb_nonzero_elements) in enumerate(zip(matrices_shapes_list[:-1], nb_elements_distribution[:-1])):
                fact_cons.append(sp(factor_shape, factor_nb_nonzero_elements, normalized=True))
                nb_residual_elements = nb_elements_distribution[factor_i + 1]
                residual_shape = (factor_shape[1], weights.shape[1])
                res_cons.append(sp(residual_shape, nb_residual_elements, normalized=True))

            # Set stopping criteria
            local_stopping_criteria = StoppingCriterion()
            global_stopping_criteria = StoppingCriterion()

            param = ParamsHierarchical(fact_cons,
                                    res_cons,
                                    local_stopping_criteria,
                                    global_stopping_criteria)

            approximation = hierarchical(weights, param, backend=2016)
            sparse_factors = [approximation.factors(i) for i in range(self.nb_matrices)]

            res_dict = dict()
            res_dict["type"] = "HierarchicalFaust"
            res_dict["faust_approximation"] = sparse_factors
            res_dict["approx_mat_dense"] = self.faust_factorization_to_dense(sparse_factors, nb_nonzero_elements_list=nb_elements_distribution)
            res_dict["nb_parameters"] = sum(nb_elements_distribution)
            return res_dict
        except:
            res_dict = dict()
            res_dict["type"] = "HierarchicalFaust"
            res_dict["faust_approximation"] = None
            res_dict["approx_mat_dense"] = np.zeros_like(weights)
            res_dict["nb_parameters"] = 0
            return res_dict
        
if __name__ == "__main__":
    # Test case
    model_filename = "data/30hiddenneurons/model.json"
    net = load_network_json(model_filename)

    target_mat = net[2].weight.detach().numpy().astype("float64")

    approximator = FaustApproximator(nb_matrices=7, linear_nb_nonzero_elements_distribution=True)
    res_dict = approximator.benchmark_approximate(target_mat=target_mat, param_share=0.3, log_filepath="tmp.log")

    norm_difference = np.linalg.norm(target_mat - res_dict["approx_mat_dense"], ord="fro")
    print(norm_difference)
    print("Norm of the original matrix: " + str(np.linalg.norm(target_mat, ord="fro")))
    halt = 1