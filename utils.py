import numpy as np

from time import time


def timeit(method):
    def timed(*args, **kwargs):
        start = time()
        result = method(*args, **kwargs)
        end = time()
        return result, end-start
    return timed


def average(method):
	def averaged(*args, **kwargs):
		av_results = []
		n_trials = kwargs['n_trials']
		for trial in range(n_trials):
			if trial == 0:
				results = method(*args, **kwargs)
				for result in results:
					av_results.append(1./n_trials*result)
			else:
				results = method(*args, **kwargs)
				for ii, result in enumerate(results):
					av_results[ii] += 1./n_trials * result
		return av_results
	return averaged


def compute_nu(sigma, de, d):
    def compute_de(nu):
        return (sigma**2 / (sigma**2 + nu**2)).sum() / ( sigma[0]**2 / (sigma[0]**2 + nu**2) )
    nu = 1.
    de_ = compute_de(nu)
    while de_ >= de:
        nu = 2*nu
        de_ = compute_de(nu)
    # binary search
    nu_max = nu 
    nu_min = 0.
    nu = (nu_max + nu_min)/2
    de_ = compute_de(nu)
    while np.abs(de_-de) > 1e-4:
        if de_ > de:
            nu_min = nu
            nu = (nu_min + nu_max) / 2
            de_ = compute_de(nu)
        else:
            nu_max = nu
            nu = (nu_min + nu_max) / 2
            de_ = compute_de(nu)
    return max(1e-10, nu)


def generate_example(n=2**12, d=2**10, de=100, c=1):
    A = np.random.randn(n, d)
    u, _, vh = np.linalg.svd(A, full_matrices=False)
    sigma = np.array([0.99**(jj) for jj in range(d)])
    A = u @ np.diag(sigma) @ vh 
    nu = compute_nu(sigma, de, d)
    xpl = 1./np.sqrt(d) * np.random.randn(d, c)
    b = A @ xpl + 1./np.sqrt(n) * np.random.randn(n, c)
    condition_number = (sigma[0]**2 + nu**2) / (sigma[-1]**2 + nu**2)
    return A, b, nu, condition_number



def sketch_size_bound(sketch, n, d, rho):
    if sketch == 'gaussian':
        return d/rho 
    elif sketch == 'srht':
        return (d + np.log(n)*np.log(d))/rho 
    elif sketch == 'sjlt':
        return d**2 / rho 
    elif sketch == 'sjlt_dense':
        return d*np.log(d) / rho 
    else:
        raise NotImplementedError


def invert_rho(rho, accelerated=False, eps=5e-3):
    def cvg_rate(z):
        kappa = (1+np.sqrt(z)) / (1-np.sqrt(z))
        if accelerated:
            return kappa * (np.sqrt(kappa)-1)**2 / (np.sqrt(kappa)+1)**2
        else:
            return kappa * (kappa-1)**2 / (kappa+1)**2
    z_min = 0
    z_max = rho
    while True:
        z = (z_min+z_max)/2
        rho_ = cvg_rate(z)
        if np.abs(rho_-rho) <= eps:
            return z
        if rho_ > rho:
            z_max = z
        else:
            z_min = z
