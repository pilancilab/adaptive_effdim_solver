import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler 
from sklearn.kernel_approximation import RBFSampler

import argparse
from os import path 

from solvers import *
from utils import generate_example


def argparser():
	parser = argparse.ArgumentParser()
	parser.add_argument('--dataset', type=str, default='synthetic')
	parser.add_argument('--n', type=int, default=2048)
	parser.add_argument('--d', type=int, default=400)
	parser.add_argument('--c', type=int, default=1)
	parser.add_argument('--de', type=int, default=300)
	parser.add_argument('--nu', type=float, default=.1)

	parser.add_argument('--rho', type=float, default=0.95)

	parser.add_argument('--n_iter_cg', type=int, default=100)
	parser.add_argument('--n_iter', type=int, default=25)
	parser.add_argument('--n_trials', type=int, default=1)
	parser.add_argument('--time_factor', type=float, default=10)

	parser.add_argument('--cholesky', type=int, default=1)

	parser.add_argument('--res_dir', default='./results/')
	
	parser.add_argument('--compare_solvers', default=1)

	return parser.parse_args()



def main(args):

	time_factor, n_trials, n_iter_cg, n_iter = args.time_factor, args.n_trials, args.n_iter_cg, args.n_iter

	print('-----------------------------------------------')
	print('Generating/loading ' + args.dataset + ' dataset')

	if args.dataset == 'synthetic':
		n, d, c, de = int(args.n), int(args.d), int(args.c), int(args.de)
		A, b, nu, kappa = generate_example(n=n, d=d, c=c, de=de)
	else:
		if args.dataset == 'wesad':
			df = pd.read_csv('data/may14_feats4.csv', index_col=0)
			A = df.drop('label', axis=1).values
			b = df['label'].values
			sc = StandardScaler()
			A = sc.fit_transform(A)
			rbf_feature = RBFSampler(gamma=0.01, n_components=5000, random_state=1)
			A = rbf_feature.fit_transform(A)
			b = b.reshape((-1,1))  
		else:
			data_ = np.load('./datasets/' + args.dataset + '.npz')
			data = {key: data_[key].squeeze() for key in data_.files}
			A, b, sigma = data['A'], data['b'], data['sigma']

		n, d = A.shape
		c = b.shape[1]
		nu = args.nu 
		de = int( (sigma**2 / (sigma**2 + nu**2)).sum() / (sigma[0]**2 / (sigma[0]**2 + nu**2)) )
		kappa = ((sigma[0]**2 + nu**2) / (sigma[-1]**2 + nu**2))
	print('dimensions n, d', n, d)
	print('--- effective dimension', de, ' --- condition number', kappa)
	
	if args.compare_solvers == 1:
	    print('-----------------')
	    print('Comparing solvers')
	    errs, times, sketch_sizes = {}, {}, {}
	    if args.cholesky == 1:
	        print('--- Direct method with Cholesky decomposition')
	        dm = directMethod(A, b, nu)
	        _, b_time = dm.solve()
	        x_opt = dm.x_opt
	        times['baseline_time'] = b_time
	    else:
	        x_opt = None
	        b_time = np.inf
	        times['baseline_time'] = b_time
	    print('--- Conjugate gradient method')
	    cg = CG(A, b, nu, x_opt=x_opt, baseline_time=b_time)
	    _, errs_cg, times_cg = cg.solve(n_iterations=n_iter_cg, time_factor=time_factor, n_trials=n_trials)
	    errs['cg'], times['cg'] = errs_cg, times_cg
	    
	    print('--- Pre-conditioned CG with srht')
	    pcg = precondCG(A, b, nu, x_opt=x_opt, baseline_time=b_time)
	    _, errs_pcg_srht, times_pcg_srht = pcg.solve(sketch_size=int(2*d), sketch='srht', n_iterations=n_iter, time_factor=time_factor, n_trials=n_trials)
	    errs['pcg_srht'], times['pcg_srht'] = errs_pcg_srht, times_pcg_srht
	    print('--- Pre-conditioned CG with sjlt')
	    pcg = precondCG(A, b, nu, x_opt=x_opt, baseline_time=b_time)
	    _, errs_pcg_sjlt, times_pcg_sjlt = pcg.solve(sketch_size=int(2*d), sketch='sjlt', n_iterations=n_iter, time_factor=time_factor, n_trials=n_trials)
	    errs['pcg_sjlt'], times['pcg_sjlt'] = errs_pcg_sjlt, times_pcg_sjlt
	    
	    m_init = 512
	    print('Adaptive Methods with initial sketch size:', m_init)
	    
	    print('--- Adaptive IHS with srht')
	    ada_ihs = adaptiveIHS(A, b, nu, x_opt=x_opt, baseline_time=b_time)
	    _, errs_ihs_srht, times_ihs_srht, m_ihs_srht = ada_ihs.solve(m_initial=m_init, rho=args.rho, sketch='srht', n_iterations=n_iter, n_trials=n_trials, time_factor=time_factor)
	    errs['adaihs_srht'], times['adaihs_srht'], sketch_sizes['adaihs_srht'] = errs_ihs_srht, times_ihs_srht, m_ihs_srht
	    
	    print('--- Adaptive IHS with sjlt')
	    ada_ihs = adaptiveIHS(A, b, nu, x_opt=x_opt, baseline_time=b_time)
	    _, errs_ihs_sjlt, times_ihs_sjlt, m_ihs_sjlt = ada_ihs.solve(m_initial=m_init, rho=args.rho, sketch='sjlt', n_iterations=n_iter, n_trials=n_trials, time_factor=time_factor)
	    errs['adaihs_sjlt'], times['adaihs_sjlt'], sketch_sizes['adaihs_sjlt'] = errs_ihs_sjlt, times_ihs_sjlt, m_ihs_sjlt
	    
	    print('--- Adaptive CG with srht')
	    ada_cg = adaptiveCG(A, b, nu, x_opt=x_opt, baseline_time=b_time)
	    _, errs_adacg_srht, times_adacg_srht, m_adacg_srht = ada_cg.solve(m_initial=m_init, rho=args.rho, sketch='srht', n_iterations=n_iter, time_factor=time_factor, n_trials=n_trials)
	    errs['adacg_srht'], times['adacg_srht'], sketch_sizes['adacg_srht'] = errs_adacg_srht, times_adacg_srht, m_adacg_srht
	    
	    print('--- Adaptive CG with sjlt')
	    ada_cg = adaptiveCG(A, b, nu, x_opt=x_opt, baseline_time=b_time)
	    _, errs_adacg_sjlt, times_adacg_sjlt, m_adacg_sjlt = ada_cg.solve(m_initial=m_init, rho=args.rho, sketch='sjlt', n_iterations=n_iter, time_factor=time_factor, n_trials=n_trials)
	    errs['adacg_sjlt'], times['adacg_sjlt'], sketch_sizes['adacg_sjlt'] = errs_adacg_sjlt, times_adacg_sjlt, m_adacg_sjlt
	    
	    print('--------------')
	    print('Saving results')
	    file_specs = args.dataset + '_nu_' + str(np.around(nu, decimals=5))
	    if args.dataset == 'synthetic':
	        file_specs += '_n_' + str(n) + '_d_' + str(d) + '_c_' + str(c) + '_de_' + str(de)
	    np.savez(args.res_dir + file_specs + '.npz', dims=np.array([n, d, de, nu, kappa, args.rho]), errs=errs, times=times, sketch_sizes=sketch_sizes)
	    print(args.res_dir + file_specs + '.npz')
	

if __name__ == '__main__':
	args = argparser()
	main(args)
