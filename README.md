# Sketching-based preconditioned iterative solvers with adaptive sketch size at the level of the effective dimension

Optimization problem
```math
\min_{x \in \mathbb{R}^d} \left\{ f(x) := \frac{1}{2} \|Ax-b\|_2^2 + \frac{\nu^2}{2} \|x\|_2^2 \right\}
```

Example: iterative Hessian sketch.
$$x_{t+1} = x_t - \mu_t H_S^{-1} \nabla f(x_t)$$
where $$H_S = A^\top S^\top S A + \nu^2 I_d$$ and $$S \in \mathbb{R}^{m \times n}$$ sketching matrix (Gaussian, SRHT, SJLT).

Choice of sketch size $$m$$? Can be as small as the effective dimension $$d_e \approx \sum_{i=1}^d \frac{\sigma_i^2}{\sigma_i^2 + \nu^2}$$, where $$\sigma_1 \geq \dots \geq \sigma_d$$ are the singular values of $$A$$.

Problem: computing $$d_e$$ is in general expensive.

Adaptive methods: time-varying (adaptive) sketch size $$m_t$$; start with $$m_0 = 1$$; at each step, check whether enough progress is made; if not, double sketch size $$m_t$$ and recompute $$H_{S_t}$$.

Adaptive iterative methods: iterative Hessian sketch, preconditioned conjugate gradient method, heavy-ball method.

See [1] for details.

## Requirements

Numpy, Scipy, Scikit-learn, Pandas, cvxopt

## Usage

### Data

Download datasets from openml.org, e.g, Dilbert
```
python ./datasets/dowload_dataset_openml --dataset 'dilbert'
```
Download WESAD dataset [2]
```
chmod u+x download_wesad.sh
./datasets/download_wesad.sh
```

### Compare iterative solvers
```
python main.py --dataset 'dilbert' --nu 1 --n_trials 1
```
Results stored in ./results/dilbert_nu_1.npz

### Plot results
Instructions in jupyter notebook ./plot_results.ipynb

## References
[1] J. Lacotte and M. Pilanci. Effective dimension adaptive sketching methods for faster regu-856larized least-squares optimization. NeurIPS 2020.
[2] P. Schmidt, A. Reiss, R. Duerichen, C. Marberger, and K. Van Laerhoven. Introducing WESAD, a multimodal dataset for wearable stress and affect detection. In ICMI, 2018.
