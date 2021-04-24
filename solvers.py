import numpy as np 
from scipy.linalg import cholesky, solve_triangular
from scipy.sparse.linalg import lsqr

from quadratic import *
from sketches import SKETCH_FN

from utils import timeit, average, sketch_size_bound, invert_rho



class directMethod(Quadratic):
	"""
	Solves exactly a quadratic optimization problem min_x {1/2 * <X*x, X*x> - <y,x> + nu^2/2 <x,x>} with a matrix decomposition method.

	Parameters
	----------
	linear_operator: user-defined linear operator with shape (n,d) which represents the matrix X
	linear_term: np.ndarray with shape (d,) or (d,c) which represents the linear term y
	reg_parameter: np.float which represents the regularization parameter nu
	"""

	def __init__(self, linear_operator, linear_term, reg_parameter, ridge_regression=True):
		
		Quadratic.__init__(self, linear_operator, linear_term, reg_parameter, ridge_regression)


	@timeit
	def solve(self, solver='cholesky'):
		"""
		Computes optimal solution of quadratic optimization problem based on Cholesky or Singular Value Decomposition
		of the linear operator.

		Parameters
		----------
		solver: {'cholesky', 'svd'}

		Notes
		-----
		This method converts the linear operator into an np.ndarray with shape (n,d)
		and applies matrix decomposition methods (Cholesky or SVD) from the scipy library
		to compute the optimal solution. This may be highly inefficient. 
		This method is implemented for the purpose of comparison with iterative solvers.
		"""

		if hasattr(self.X, 'row_slice'):
			X_ = self.X.row_slice(range(self.X.shape[0]))
		else:
			X_ = self.X.mul(np.eye(self.d))

		if solver == 'cholesky':
			hessian = X_.T @ X_ + self.nu**2 * np.eye(self.d)
			L = cholesky(hessian)
			self.x_opt = solve_triangular(L, solve_triangular(L.T, self.y, lower=True))
		elif solver == 'svd':
			_, sigma, vh = np.linalg.svd(X_, full_matrices=False)
			self.x_opt = vh.T @ ((1/(sigma**2 + self.nu**2)).reshape((-1,1)) * (vh @ self.y))
		else:
			raise NotImplementedError




class Solver(Quadratic):

	def __init__(self, linear_operator, linear_term, reg_parameter, ridge_regression=True, x_opt=None, baseline_time=np.inf):

		Quadratic.__init__(self, linear_operator, linear_term, reg_parameter, 
			               ridge_regression=ridge_regression, x_opt=x_opt, baseline_time=baseline_time)


	def hess_vec_product(self, x):
		"""
		Given a vector x, computes the Hessian-vector product H * x, where H = X' * X + nu^2 * I_d

		Parameters
		----------
		x: np.ndarray with shape (d,) or (d,1)
		"""
		if x.ndim == 1:
			x = x.reshape((-1,1))

		return self.X.hmul(self.X.mul(x)) + self.nu**2 * x



	def factor_approx_hessian(self, sketch_fn, sketch_size):
		"""
		Form and factorize the approximate Hessian X'S'SX + nu^2 * I, based on an (m,n) embedding matrix S.

		Parameters
		----------
		sketch_fn: function which takes as inputs a LinearOperator instance with shape (n,d) and a np.int sketch size, and returns an np.ndarray with shape (sketch size, n)
		sketch_size: np.int, represents the sketch size m.
	
		Notes
		-----
		We distinguish the cases m >= d and m < d to minimize the cost of computing either (SX)'*SX or SX*(SX)'.
		If m >= d, it forms (SX)' * SX + nu^2 I_d and then computes its Cholesky decomposition.
		If m < d, it forms SX * (SX)' + nu^2 I_m and then computes its Cholesky decomposition.
		The Cholesky decomposition is then primarily used for solving linear systems of the form (SX' * SX + nu^2 * I_d) * x = g.
		"""

		self.SX = sketch_fn(self.X, sketch_size)

		if sketch_size > self.d:
			self.U = cholesky(self.SX.T @ self.SX + self.nu**2 * np.eye(self.d))
		else:
			self.U = cholesky(self.SX @ self.SX.T + self.nu**2 * np.eye(sketch_size))



	def solve_approx_newton_system(self, z):
		"""
		Solves linear system ((SX)' * SX + nu^2 * I_d) * x = z.

		Parameters
		----------
		z: np.ndarray with shape (m,) or (m,1), where m is the sketch size

		Returns
		-------
		x: np.ndarray with shape (d,1); exact solution of ((SX)' * SX + nu^2 * I_d) * x = z.

		Notes
		-----
		We distinguish the cases m >= d and m < d in order to minimize the costs of the matrix multiplication (either (SX)' * SX or SX * (SX)').
		If m >= d, the linear system is solved based on the Cholesky decomposition of (SX)' * SX + nu^2 I_d.
		If m < d, the linear system is solved based on the Cholesky decomposition of SX * (SX)' + nu^2 I_m. In this case, we leverage the Woodbury matrix identity to perform the inversion.
		"""

		if z.ndim == 1:
			z = z.reshape((-1,1))

		sketch_size = self.SX.shape[0]

		if sketch_size >= self.d:
			return solve_triangular(self.U, solve_triangular(self.U.T, z, lower=True))
		else:
			return 1./self.nu**2*(z - self.SX.T @ solve_triangular(self.U, solve_triangular(self.U.T, self.SX @ z, lower=True)))




class CG(Solver):
	"""
	Solves a quadratic optimization problem min_x 1/2 * |X*x|^2 - y' * x + nu^2/2 |x|^2 using the conjugate gradient method.

	Parameters
	----------
	linear_operator: user-defined linear operator with shape (n,d) which represents the matrix X
	linear_term: np.ndarray with shape (n,) or (n,1) which represents the vector y
	reg_parameter: np.float which represents the regularization parameter nu
	"""

	def __init__(self, linear_operator, linear_term, reg_parameter, ridge_regression=True, x_opt=None, baseline_time=np.inf):

		Solver.__init__(self, linear_operator, linear_term, reg_parameter, 
						ridge_regression=ridge_regression, x_opt=x_opt, baseline_time=baseline_time)


	@timeit 
	def __init_cg(self):
		x = np.zeros((self.d, self.c))
		r, p = self.y.copy(), self.y.copy()
		return x, r, p


	@timeit
	def __cg_iteration(self, x, r, p):
		Hp = self.hess_vec_product(p)
		alpha = np.sum(r**2, axis=0) / np.sum(p*Hp, axis=0)
		x = x + alpha*p 
		_r = r - alpha*Hp 
		beta = np.sum(_r**2, axis=0) / np.sum(r**2, axis=0)
		r = np.copy(_r)
		p = r + beta*p 
		return x, r, p 


	@average
	def solve(self, n_iterations, time_factor=np.inf, n_trials=1):
		(x, r, p), time_ = self.__init_cg()
		errors = [self.compute_error(x)]
		times = [time_]; tt_time = time_
		iteration = 0
		while (iteration < n_iterations) and (tt_time < time_factor * self.baseline_time):
			(x, r, p), time_ = self.__cg_iteration(x, r, p)
			iteration += 1
			tt_time += time_
			errors.append(self.compute_error(x))
			times.append(tt_time)
		return x, np.array(errors), np.array(times)




class precondCG(Solver):

	def __init__(self, linear_operator, linear_term, reg_parameter, ridge_regression=True, x_opt=None, baseline_time=np.inf):

		Solver.__init__(self, linear_operator, linear_term, reg_parameter, 
						ridge_regression=ridge_regression, x_opt=x_opt, baseline_time=baseline_time)


	@timeit 
	def __init_pcg(self, sketch_fn, sketch_size):
		self.factor_approx_hessian(sketch_fn, sketch_size)
		x = 1./np.sqrt(self.d)*np.random.randn(self.d, self.c)
		r = self.y - self.hess_vec_product(x) 
		p = self.solve_approx_newton_system(r)
		rtilde = p.copy()
		delta = np.sum(r*rtilde, axis=0)
		return x, r, p, rtilde, delta


	@timeit 
	def __pcg_iteration(self, x, r, p, rtilde, delta):
		Hp = self.hess_vec_product(p)
		alpha = delta / np.sum(p*Hp, axis=0)
		x = x + alpha * p 
		r = r - alpha * Hp 
		rtilde = self.solve_approx_newton_system(r)
		delta_ = np.sum(r*rtilde, axis=0)
		beta = delta_ / delta 
		delta = delta_ 
		p = rtilde + beta*p 
		return x, r, p, rtilde, delta 


	@average
	def solve(self, sketch_size, n_iterations, sketch='srht', time_factor=np.inf, n_trials=1):

		sketch_fn = SKETCH_FN[sketch]

		(x, r, p, rtilde, delta), time_ = self.__init_pcg(sketch_fn, sketch_size)

		errors = [self.compute_error(x)]; tt_time = time_; times = [tt_time]; iteration = 0

		while (iteration < n_iterations) and (tt_time < time_factor * self.baseline_time):
			(x, r, p, rtilde, delta), time_ = self.__pcg_iteration(x, r, p, rtilde, delta)
			tt_time += time_
			iteration += 1
			errors.append(self.compute_error(x))
			times.append(tt_time)
		return x, np.array(errors), np.array(times)



class adaptiveIHS(Solver):

	def __init__(self, linear_operator, linear_term, reg_parameter, ridge_regression=True, x_opt=None, baseline_time=np.inf):

		Solver.__init__(self, linear_operator, linear_term, reg_parameter, 
						ridge_regression=ridge_regression, x_opt=x_opt, baseline_time=baseline_time)


	@timeit 
	def __init_ihs(self, x, sketch_fn, sketch_size):
		self.factor_approx_hessian(sketch_fn, sketch_size)
		g = self.hess_vec_product(x) - self.y
		v = self.solve_approx_newton_system(g)
		delta = 1/2 * np.sum(g * v, axis=0)
		return v, delta 


	@timeit 
	def __ihs_iteration(self, x, v, delta):
		x_new = x - self.mu * v 
		g_new = self.hess_vec_product(x_new) - self.y
		v_new = self.solve_approx_newton_system(g_new)
		delta_new = 1/2 * np.sum(g_new*v_new, axis=0)
		return x_new, v_new, delta_new


	@average
	def solve(self, n_iterations, m_initial=1, rho=0.9, sketch='srht', time_factor=np.inf, n_trials=1):

		rho_approx = invert_rho(rho)
		sketch_fn = SKETCH_FN[sketch]
		sk_bound = min(self.n-1, sketch_size_bound(sketch, self.n, self.d, rho_approx))
		self.mu = 1-rho_approx

		if sketch == 'gaussian':
			rho_gaussian = -1 + 2/rho_approx * (1-np.sqrt(1-rho_approx))
			self.mu = (1-rho_gaussian)**2 / (1+rho_gaussian)
			sk_bound = min(self.n-1, sketch_size_bound(sketch, self.n, self.d, rho_gaussian))
			
		sketch_size = int(m_initial)
		x = 1./np.sqrt(self.d) * np.random.randn(self.d, self.c)

		(v, delta), time_ = self.__init_ihs(x, sketch_fn, sketch_size)
		errors, times, sketch_sizes = [self.compute_error(x)], [time_], [sketch_size]
		
		iteration = 0; tt_time = time_
		while (iteration < n_iterations) and (tt_time < time_factor * self.baseline_time):
			(x_, v_, delta_), time_ = self.__ihs_iteration(x, v, delta)
			tt_time += time_
			if np.any(delta_/delta > rho) and sketch_size <= sk_bound:
				sketch_size = int(2 * sketch_size)
				(v, delta), time_ = self.__init_ihs(x, sketch_fn, sketch_size)
				tt_time += time_
			else:
				iteration += 1
				x, v, delta = x_.copy(), v_.copy(), delta_.copy()
				errors.append(self.compute_error(x))
				times.append(tt_time)
				sketch_sizes.append(sketch_size)

		return x, np.array(errors), np.array(times), np.array(sketch_sizes)



class adaptiveCG(Solver):

	def __init__(self, linear_operator, linear_term, reg_parameter, ridge_regression=True, x_opt=None, baseline_time=np.inf):

		Solver.__init__(self, linear_operator, linear_term, reg_parameter, 
						ridge_regression=ridge_regression, x_opt=x_opt, baseline_time=baseline_time)


	@timeit 
	def __init_pcg(self, x, sketch_fn, sketch_size):
		self.factor_approx_hessian(sketch_fn, sketch_size)
		r = self.y - self.hess_vec_product(x) 
		rtilde = self.solve_approx_newton_system(r)
		delta = np.sum(r*rtilde, axis=0)
		p = rtilde.copy()
		return r, p, rtilde, delta


	@timeit 
	def __pcg_iteration(self, x, r, p, rtilde, delta):
		Hp = self.hess_vec_product(p)
		alpha = delta / np.sum(p*Hp, axis=0)
		x_new = x + alpha*p 
		r_new = r - alpha*Hp 
		rtilde_new = self.solve_approx_newton_system(r_new)
		delta_new = np.sum(r_new*rtilde_new, axis=0) 
		p_new = rtilde_new + (delta_new/delta)*p
		return x_new, r_new, p_new, rtilde_new, delta_new


	@average
	def solve(self, n_iterations, m_initial=1, rho=0.9, sketch='srht', time_factor=np.inf, n_trials=1):
		
		rho_approx = invert_rho(rho, accelerated=True)
		sketch_fn = SKETCH_FN[sketch]
		sk_bound = min(self.n-1, sketch_size_bound(sketch, self.n, self.d, rho_approx))
		
		sketch_size = int(m_initial)
		x = 1./np.sqrt(self.d)*np.random.randn(self.d, self.c)

		(r, p, rtilde, delta), time_ = self.__init_pcg(x, sketch_fn, sketch_size)
		errors, times, sketch_sizes = [self.compute_error(x)], [time_], [sketch_size]

		iteration = 0; tt_time = time_
		while (iteration < n_iterations) and (tt_time < time_factor * self.baseline_time):
			(x_, r_, p_, rtilde_, delta_), time_ = self.__pcg_iteration(x, r, p, rtilde, delta)
			tt_time += time_ 
			if np.any(delta_ / delta > rho) and sketch_size <= sk_bound:
				sketch_size = int(2*sketch_size)
				(r, p, rtilde, delta), time_ = self.__init_pcg(x, sketch_fn, sketch_size)
				tt_time += time_
			else:
				iteration += 1
				x, r, p, rtilde, delta = x_.copy(), r_.copy(), p_.copy(), rtilde_.copy(), delta_.copy()
				errors.append(self.compute_error(x))
				times.append(tt_time)
				sketch_sizes.append(sketch_size)
		return x, np.array(errors), np.array(times), np.array(sketch_sizes)





def test_dm():
	"""
	Test direct method for overdetermined ridge regression problem
	"""
	n, d, c, nu = 16, 8, 2, 0.5
	A = np.random.randn(n,d)
	b = np.random.randn(n,c)
	x_opt = np.linalg.inv(A.T @ A + nu**2 * np.eye(d)) @ (A.T @ b)
	dm = directMethod(A, b, nu, ridge_regression=True)
	_ = dm.solve()
	assert np.linalg.norm(dm.x_opt - x_opt) < 1e-10
	"""
	Test direct method for underdetermined ridge regression problem
	"""
	n, d, c, nu = 8, 16, 2, 0.5
	A = np.random.randn(n,d)
	b = np.random.randn(n,c)
	z_opt = np.linalg.inv(A @ A.T + nu**2 * np.eye(n)) @ b
	dm = directMethod(A, b, nu, ridge_regression=True)
	_ = dm.solve()
	assert np.linalg.norm(dm.x_opt - z_opt) < 1e-10

	print('direct method test successfully passed')



def test_cg():
	"""
	Test CG for overdetermined ridge regression problem
	"""
	n, d, c, nu = 16, 8, 2, 0.5
	A = np.random.randn(n,d)
	b = np.random.randn(n,c)
	x_opt = np.linalg.inv(A.T @ A + nu**2 * np.eye(d)) @ (A.T @ b)
	cg = CG(A, b, nu, ridge_regression=True, x_opt=x_opt)
	x_cg, _, _ = cg.solve(n_iterations=20, n_trials=1)
	print('CG test passed with final error', np.linalg.norm(A @ (x_cg - x_opt))**2)
	"""
	Test CG for underdetermined ridge regression problem
	"""
	n, d, c, nu = 8, 16, 2, 0.5
	A = np.random.randn(n,d)
	b = np.random.randn(n,c)
	z_opt = np.linalg.inv(A @ A.T + nu**2 * np.eye(n)) @ b
	cg = CG(A, b, nu, ridge_regression=True, x_opt=z_opt)
	z_cg, _, _ = cg.solve(n_iterations=20, n_trials=1)
	print('CG test passed with final error', np.linalg.norm(A.T @ (z_cg - z_opt))**2)



def test_pcg(sketch='srht'):
	"""
	Test preconditioned CG for overdetermined ridge regression problem
	"""
	n, d, c, nu = 510, 64, 2, 0.5
	A = np.random.randn(n,d)
	b = np.random.randn(n,c)
	x_opt = np.linalg.inv(A.T @ A + nu**2 * np.eye(d)) @ (A.T @ b)
	pcg = precondCG(A, b, nu, ridge_regression=True, x_opt=x_opt)
	x_pcg, _, _ = pcg.solve(sketch=sketch, sketch_size=int(4*d), n_iterations=20, n_trials=1)
	print('pCG test with ' + sketch + ' passed with final error ', np.linalg.norm(A @ (x_pcg - x_opt))**2)
	"""
	Test preconditioned CG for underdetermined ridge regression problem
	"""
	n, d, c, nu = 64, 510, 2, 0.5
	A = np.random.randn(n,d)
	b = np.random.randn(n,c)
	z_opt = np.linalg.inv(A @ A.T + nu**2 * np.eye(n)) @ b
	pcg = precondCG(A, b, nu, ridge_regression=True, x_opt=z_opt)
	z_pcg, _, _ = pcg.solve(sketch=sketch, sketch_size=int(4*n), n_iterations=20, n_trials=1)
	print('pCG test with ' + sketch + ' passed with final error ', np.linalg.norm(A.T @ (z_pcg - z_opt))**2)



def test_adaihs(sketch='srht'):
	"""
	Test adaptive IHS for overdetermined ridge regression problem
	"""
	n, d, c, nu = 1024, 32, 1, 1.
	A = np.random.randn(n,d)
	b = np.random.randn(n,c)
	x_opt = np.linalg.inv(A.T @ A + nu**2 * np.eye(d)) @ (A.T @ b)
	adaihs = adaptiveIHS(A, b, nu, ridge_regression=True, x_opt=x_opt)
	x_ihs, _, _, _ = adaihs.solve(sketch=sketch, m_initial=256, n_iterations=40, n_trials=1)
	print('adaIHS test with ' + sketch + ' passed with final error ', np.linalg.norm(A @ (x_ihs - x_opt))**2 )
	"""
	Test adaptive IHS for underdetermined ridge regression problem
	"""
	n, d, c, nu = 32, 1024, 1, 1.
	A = np.random.randn(n,d)
	b = np.random.randn(n,c)
	z_opt = np.linalg.inv(A @ A.T + nu**2 * np.eye(n)) @ b
	adaihs = adaptiveIHS(A, b, nu, ridge_regression=True, x_opt=z_opt)
	z_ihs, _, _, _ = adaihs.solve(sketch=sketch, m_initial=256, n_iterations=40, n_trials=1)
	print('adaIHS test with ' + sketch + ' passed with final error ', np.linalg.norm(A.T @ (z_ihs - z_opt))**2)



def test_adacg(sketch='srht'):
	"""
	Test adaptive CG for overdetermined ridge regression problem
	"""
	n, d, c, nu = 1024, 32, 1, 1.
	A = np.random.randn(n,d)
	b = np.random.randn(n,c)
	x_opt = np.linalg.inv(A.T @ A + nu**2 * np.eye(d)) @ (A.T @ b)
	adacg = adaptiveCG(A, b, nu, ridge_regression=True, x_opt=x_opt)
	x_adacg, _, _, _ = adacg.solve(sketch=sketch, m_initial=256, n_iterations=20, n_trials=1)
	print('adaCG test with ' + sketch + ' passed with final error ', np.linalg.norm(A @ (x_adacg - x_opt))**2 )
	"""
	Test adaptive CG for underdetermined ridge regression problem
	"""
	n, d, c, nu = 32, 1024, 1, 1.
	A = np.random.randn(n,d)
	b = np.random.randn(n,c)
	z_opt = np.linalg.inv(A @ A.T + nu**2 * np.eye(n)) @ b
	adacg = adaptiveCG(A, b, nu, ridge_regression=True, x_opt=z_opt)
	z_adacg, _, _, _ = adacg.solve(sketch=sketch, m_initial=256, n_iterations=20, n_trials=1)
	print('adaCG test with ' + sketch + ' passed with final error ', np.linalg.norm(A.T @ (z_adacg - z_opt))**2)