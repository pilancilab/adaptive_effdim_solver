# Sketching-based preconditioned iterative solvers with adaptive sketch size 

## Requirements

Numpy, Scipy, Scikit-learn, Pandas, Cvxopt

## Usage

### Data

Download datasets from openml.org, e.g, Dilbert
```
python ./datasets/dowload_dataset_openml --dataset 'dilbert'
```
Download WESAD dataset [1]
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
