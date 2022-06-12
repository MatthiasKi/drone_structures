import numpy as np
import os
from phoenix_drone_simulation.utils.trajectory_generator import TrajectoryGenerator
from phoenix_drone_simulation.utils.utils import load_network_json, dump_network_json, get_file_contents
import torch.nn as nn
import torch
import copy
import pickle

from faust_approximator import FaustApproximator
from sparse_approximation import SparseApproximator
from low_rank_approximation import LowRankApproximator
from semiseparable_approximation import SSSApproximator
from td_ldr_approximation import TDLDRApproximator
from heuristic_low_rank_matrix_norm_approximation import LowRankMatrixNormApproximator

# Hyperparameters
model_filename = "models/30hiddenneurons.json"
output_folder_name = "benchmark/"
param_shares = [0.7]
approximators = [
    SparseApproximator(),
    LowRankApproximator(),
    SSSApproximator(),
    # NOTE that the LDR approximators work only for square matrices (for all other matrices it will return a zero matrix)
    TDLDRApproximator(norm="fro"),
    LowRankMatrixNormApproximator(norm="nuc"),
    FaustApproximator(nb_matrices=2, linear_nb_nonzero_elements_distribution=True),
]
nb_test_trajectories = 50
target_mat_layer_indices = [2] # If set to None, then all linear layers are appproximated
# ---

def log(s: str):
    with open(os.path.join(output_folder_name, "params.txt"), "a") as f:
        f.write(s + "\n")

def get_modified_net(net: torch.nn.Module, approximated_matrices: list, layer_indices: list):
    modified_net = copy.deepcopy(net)
    for idx, layer_idx in enumerate(layer_indices):
        weights_torch = torch.nn.Parameter(torch.tensor(approximated_matrices[idx].astype(np.float32)))
        modified_net[layer_idx].weight = weights_torch
    return modified_net

def get_verbosity_logger_filepath(base_path, approximator, param_share):
    log_filename = "log_" + type(approximator).__name__ + "_" + str(param_share) + ".txt"
    return os.path.join(base_path, log_filename)

if os.path.exists(output_folder_name):
    raise Exception("Output Folder Path alrady exists")

os.mkdir(output_folder_name)

generator = TrajectoryGenerator("DroneCircleBulletEnv-v0")
generator.load_file_from_disk(model_filename)

net = load_network_json(model_filename)
model_name = model_filename.split("/")[-1].split(".")[0]

data = get_file_contents(model_filename)
scaling_parameters = np.array(data['scaling_parameters'])
activation_function = data['activation']

generator.policy_net = net
base_reward, base_reward_std = generator.evaluate(num_trajectories=nb_test_trajectories, return_stds=True)

log("Approximated model filename: " + str(model_filename))
log("Paramshares: " + str(param_shares))
log("Approx Methods: " + str([type(approximator).__name__ for approximator in approximators]))
log("Nb Test Trajectories: " + str(nb_test_trajectories))
log("Base Reward: " + str(base_reward) + "+-" + str(base_reward_std))
log("Target Mat layer indices: " + str(target_mat_layer_indices))

target_mats = []
if target_mat_layer_indices is None:
    target_mat_layer_indices = []
    for layer_i, layer in enumerate(net):
        if isinstance(layer, nn.Linear):
            target_mats.append(layer.weight.data.detach().numpy().astype("float64"))
            target_mat_layer_indices.append(layer_i)
else:
    for layer_i in target_mat_layer_indices:
        assert isinstance(net[layer_i], nn.Linear), "Trying to approximate a non-linear layer"
        target_mats.append(net[layer_i].weight.data.detach().numpy().astype("float64"))

for param_share in param_shares:
    param_share_foldername = os.path.join(output_folder_name, "param_share_" + str(param_share))
    os.mkdir(param_share_foldername)

    log("")
    log("------------------------")
    log("Param Share " + str(param_share))
    log("------------------------")
    log("")

    for approximator in approximators:
        approximator_name = type(approximator).__name__
        if isinstance(approximator, FaustApproximator):
            approximator_name += "_" + str(approximator.nb_matrices) + "_" + str(approximator.linear_nb_nonzero_elements_distribution)
        elif isinstance(approximator, LowRankMatrixNormApproximator):
            approximator_name += "_" + str(approximator.norm)
        elif isinstance(approximator, TDLDRApproximator):
            approximator_name += "_" + str(approximator.norm)
        approximator_foldername = os.path.join(param_share_foldername, approximator_name)
        os.mkdir(approximator_foldername)

        approximated_matrices = []
        res_dicts = []
        for target_mat_i, target_mat in enumerate(target_mats):
            log_filepath = get_verbosity_logger_filepath(output_folder_name, approximator, param_share)
            res_dict = approximator.benchmark_approximate(target_mat=target_mat, param_share=param_share, log_filepath=log_filepath)
            res_dicts.append(res_dict)
            approximated_matrices.append(res_dict["approx_mat_dense"])

        modified_net = get_modified_net(net=net, approximated_matrices=approximated_matrices, layer_indices=target_mat_layer_indices)
        generator.policy_net = modified_net
        reward, reward_std = generator.evaluate(num_trajectories=nb_test_trajectories, return_stds=True)

        log(approximator_name + ": " + str(reward) + "+-" + str(reward_std))
        pickle.dump(res_dicts, open(os.path.join(approximator_foldername, "res_dicts.p"), "wb"))
        dump_network_json(activation=activation_function, scaling_parameters=scaling_parameters, neural_network=modified_net, file_name_path=approximator_foldername)