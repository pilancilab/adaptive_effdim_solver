# Sketching-based preconditioned iterative solvers with adaptive sketch size at the level of the effective dimension

Solve regularized least-squares optimization problem with sketching-based preconditioned iterative solvers, e.g., iterative Hessian sketch.

Choice of sketch size? Can be as small as the effective dimension of the optimization problem.

Problem: computing the effective dimension is in general expensive.

Adaptive methods: time-varying (adaptive) sketch size; start with small sketch size; at each step, check whether enough progress is made; if not, double sketch size and recompute the sketch.

Adaptive iterative methods: iterative Hessian sketch, preconditioned conjugate gradient method, heavy-ball method.

See [1,2] for details.

## Requirements

Numpy, Scipy, Scikit-learn, Pandas, cvxopt

## Usage

### Data

Download datasets from openml.org, e.g, Dilbert
```
python ./datasets/dowload_dataset_openml --dataset 'dilbert'
```
Download WESAD dataset [3] by executing ./datasets/download_wesad.sh

### Compare iterative solvers
```
python main.py --dataset 'dilbert' --nu 1 --n_trials 1
```
Results stored in ./results/dilbert_nu_1.npz

### Plot results
Instructions in jupyter notebook ./plot_results.ipynb

## References
[1] J. Lacotte and M. Pilanci. Effective dimension adaptive sketching methods for faster regularized least-squares optimization. NeurIPS 2020.
[2] J. Lacotte and M. Pilanci. Fast Convex Quadratic Optimization Solvers with Adaptive Sketching-based Preconditioners. Arxiv preprint 2021. https://arxiv.org/pdf/2104.14101.pdf
[3] P. Schmidt, A. Reiss, R. Duerichen, C. Marberger, and K. Van Laerhoven. Introducing WESAD, a multimodal dataset for wearable stress and affect detection. In ICMI, 2018.
