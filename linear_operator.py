import numpy as np 
import pandas as pd 
from scipy.sparse import issparse



class MatrixLinearOperator:

	def __init__(self, A, row_type=False):
		""" Wraps an np.ndarray into an abstract linear operator object
		Parameters
		----------
		A : list or np.ndarray or scipy.sparse array or pd dataframe with shape (n,d)
		"""
		if isinstance(A, np.ndarray) or isinstance(A, list) or isinstance(A, pd.DataFrame):
			A = np.array(A)
		elif issparse(A):
			A = A.toarray()
		else:
			raise ValueError('A not list, scipy.sparse matrix, numpy.ndarray or pandas.DataFrame')

		try:
			A = A.astype(float)
		except:
			raise ValueError('A must have numeric values')

		if A.ndim == 1:
			if not row_type:
				A = A.reshape((-1,1))
			else:
				A = A.reshape((1,-1))
		elif A.ndim > 2:
			raise ValueError('A must have at most 2 dimensions')

		self.shape = A.shape
		self.__A = A

	def mul(self, x):
		return self.__A @ x 

	def hmul(self, z):
		return self.__A.T @ z

	def col_slice(self, col_indices):
		return self.__A[:, col_indices]

	def row_slice(self, row_indices):
		return self.__A[row_indices]




class LinearOperator:
	""" Wraps a user-defined class with following attributes and methods

		Parameters
		----------
		A : np.ndarray or user-defined object with following attributes:

			linear_operator.shape : 2-d tuple containing the shape of the linear operator

			linear_operator.mul : performs multiplication x -> A*x where A represents the linear operator
				and x is a vector or matrix; must support x ndarray

			linear_operator.hmul : performs adjoint multiplication z -> A' * z, where z is a vector or matrix; 
				must support z ndarray
	"""
	def __init__(self, A=None):
		if A is None:
			pass
		elif isinstance(A, np.ndarray):
			_matrix_lin_op = MatrixLinearOperator(A)
			self.shape = _matrix_lin_op.shape
			self.mul = _matrix_lin_op.mul 
			self.hmul = _matrix_lin_op.hmul 
			self.col_slice = _matrix_lin_op.col_slice
			self.row_slice = _matrix_lin_op.row_slice
		else:
			assert hasattr(A, 'shape'), 'missing shape attribute'
			assert hasattr(A, 'mul'), 'missing mul attribute (right multiplication)'
			assert hasattr(A, 'hmul'), 'missing hmul attribute (adjoint multiplication)'
			self.shape = A.shape 
			self.mul = A.mul 
			self.hmul = A.hmul
			if hasattr(A, 'col_slice'):
				self.col_slice = A.col_slice
			if hasattr(A, 'row_slice'):
				self.row_slice = A.row_slice


	def adjoint(self):
		"""Returns a new LinearOperator instance representing its adjoint."""
		new_lin_op = LinearOperator()
		new_lin_op.shape = (self.shape[1], self.shape[0])
		new_lin_op.mul = self.hmul 
		new_lin_op.hmul = self.mul 
		if hasattr(self, 'row_slice'):
			def new_col_slice(indices):
				return self.row_slice(indices).T 
			new_lin_op.col_slice = new_col_slice 
		if hasattr(self, 'col_slice'):
			def new_row_slice(indices):
				return self.col_slice(indices).T 
			new_lin_op.row_slice = new_row_slice
		return new_lin_op