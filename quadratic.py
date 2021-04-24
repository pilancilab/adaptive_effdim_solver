import numpy as np
from scipy.linalg import cholesky, solve_triangular
from scipy.sparse.linalg import lsqr

from time import time

from linear_operator import LinearOperator



class Quadratic:
	"""
	Instantiate a quadratic optimization problem min_x {1/2 * <X*x, X*x> - <y,x> + nu**2/2 |x|^2}
	
	Parameters
	----------
	linear_op : user-defined linear operator object (see the class LinearOperator for requirements) with shape (n,d);
				this represents the matrix X
	target: np.ndarray with shape (d,) or (d,c);
			this represents the vector y
	reg_parameter: np.float
	x_opt: optimal solution
	baseline_time: baseline time to compute x_opt
	"""

	def __init__(self, linear_operator, linear_term, reg_parameter, ridge_regression=True, x_opt=None, baseline_time=np.inf):

		if linear_term.ndim == 1:
			linear_term = linear_term.reshape((-1,1))

		linear_operator = LinearOperator(linear_operator)
		n_samples, n_features = linear_operator.shape 

		if n_samples >= n_features:
			self.X = linear_operator
			self.y = self.X.hmul(linear_term)
		else:
			self.X = linear_operator.adjoint()
			self.y = linear_term

		self.n, self.d = self.X.shape
		self.c = self.y.shape[1]
		self.nu = np.float(reg_parameter)
		
		if x_opt is not None and x_opt.ndim == 1:
			x_opt = x_opt.reshape((-1,1))
		self.x_opt = x_opt

		self.baseline_time = baseline_time



	def compute_error(self, x):
		if x.ndim == 1:
			x = x.reshape((-1,1))

		if self.x_opt is not None:
			return 1./2 * ((self.X.mul(x-self.x_opt))**2).sum() + self.nu**2/2 * ((x-self.x_opt)**2).sum()
		else:
			return 1./2 * (self.X.mul(x)**2).sum() - (self.y * x).sum() + self.nu**2/2 * (x**2).sum()




def test_quadratic():
	"""
	Test overdetermined ridge regression problem
	"""
	n_samples, n_features, n_classes, nu = 16, 4, 1, 0.5
	A = np.random.randn(n_samples, n_features)
	b = np.random.randn(n_samples,n_classes)
	ridge = Quadratic(A, b, nu, ridge_regression=True)
	assert ridge.n == n_samples and ridge.d == n_features and ridge.c == n_classes and ridge.nu == nu 
	x = np.random.randn(ridge.d)
	try:
		ridge.compute_error(x)
	except:
		raise ValueError('could not compute error')
	x_opt = np.linalg.inv(A.T @ A + nu**2 * np.eye(ridge.d)) @ (A.T @ b)
	ridge.x_opt = x_opt.reshape((-1,1))
	assert np.abs(ridge.compute_error(x_opt)) < 1e-10
	"""
	Test underdetermined ridge regression problem
	"""
	n_samples, n_features, n_classes, nu = 4, 16, 2, 0.5
	A = np.random.randn(n_samples, n_features)
	b = np.random.randn(n_samples,n_classes)
	ridge = Quadratic(A, b, nu, ridge_regression=True)
	assert ridge.n == n_features and ridge.d == n_samples and ridge.c == n_classes and ridge.nu == nu
	z = np.random.randn(ridge.d)
	try:
		ridge.compute_error(z)
	except:
		raise ValueError('could not compute error') 
	z_opt = np.linalg.inv(A @ A.T + nu**2 * np.eye(ridge.d)) @ b
	ridge.x_opt = z_opt.reshape((-1,ridge.c))
	assert np.abs(ridge.compute_error(z_opt)) < 1e-10
	x_opt = np.linalg.inv(A.T @ A + nu**2 * np.eye(ridge.n)) @ (A.T @ b)
	assert np.linalg.norm(A.T @ ridge.x_opt - x_opt) < 1e-10

	print('Quadratic test successfully passed')