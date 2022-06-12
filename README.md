# Exploiting Structures in Weight Matrices for Efficient Real-Time Drone Control with Neural Networks

This repository contains the code, which were used to run the experiments described in 

Kissel, Gronauer, Korte, Sacchetto, Diepold: Exploiting Structures in Weight Matrices for Efficient Real-Time Drone Control with Neural Networks

## File Structure

- ``drone_approximation_benchmark.py``: Script used for running benchmarks. This script imports the other scripts defining the approximators. Therefore, this script can be seen as a starting point to understand how the approximators can be used. 
- ``faust_approximator.py``: Class for approximating a matrix with products of sparse matrices.
- ``heuristic_low_rank_matrix_norm_approximation.py``: Class for approximating a matrix with a low rank matrix in a desired norm (using a gradient descent based optimization scheme).
- ``low_rank_approximation.py``: Class for approximating a matrix with a low rank matrix using a randomized singular value decomposition.
- ``semiseparable_approximation.py``: Class for approximating a matrix with a sequentially semiseparable matrix.
- ``sparse_approximation.py``: Class for approximating a matrix with a sparse matrix.
- ``td_ldr_approximation.py``: Class for approximating a matrix with a low displacement rank matrix (based on tridiagonal operator matrices). 

## Usage

We recomment using anaconda or miniconda to avoid troubles with required python packages. We used conda 4.10.1, but any higher version should be fine. Conda can be obtained from [https://docs.conda.io/en/latest/miniconda.html](https://docs.conda.io/en/latest/miniconda.html). 

Create your conda environment using

``conda create -n dronestructures python=3.7``

After activating the environment with

``conda activate dronestructures``

you can install the required packages using

``python3 -m pip install -r requirements.txt``

Then, you can start benchmarks using

``python3 drone_approximation_benchmark.py``

You can set the hyperparameters for the benchmark directly in the drone_approximation_benchmark.py file. The benchmark run will create folder "benchmark/", which contains the results of the benchmark (a log file together with the approximated models). 
