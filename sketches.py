import numpy as np 
from scipy.linalg import hadamard

from linear_operator import LinearOperator




def _srht(indices, v):
    """
    Helper function for srht.
    """
    n = v.shape[0]
    if n == 1:
        return v
    i1 = indices[indices < n//2]
    i2 = indices[indices >= n//2]
    if len(i1) == 0:
        return _srht(i2-n//2, v[:n//2,::]-v[n//2:,::])
    elif len(i2) == 0:
        return _srht(i1, v[:n//2,::]+v[n//2:,::])
    else:
        return np.vstack([_srht(i1, v[:n//2,::]+v[n//2:,::]), _srht(i2-n//2, v[:n//2,::]-v[n//2:,::])])



def srht(lin_op, m, indices=None, signs=None):
    """
    Subsampled Randomized Hadamard Transform
    
    Parameters
    ----------
    lin_op : user-defined linear operator (see LinearOperator class for requirements), with shape (n,d)
    m : sketch size, np.int

    Returns
    -------
    np.ndarray with shape (m,d)
    """
    lin_op = LinearOperator(lin_op)
    n = lin_op.shape[0]

    new_dim = 2**(np.int(np.ceil(np.log(n) / np.log(2))))

    if indices is None:
        indices = np.sort(np.random.choice(np.arange(new_dim), m, replace=False))
    if signs is None:
        signs = np.random.choice([-1,1], n, replace=True).reshape((-1,1))

    if hasattr(lin_op, 'row_slice'):
        matrix = signs * lin_op.row_slice(range(n)).copy()
        if n & (n-1) != 0:
            matrix = np.vstack([matrix, np.zeros((new_dim - n, matrix.shape[1]))])
        return 1./np.sqrt(m) * _srht(indices, matrix)
    else:
        RHD = 1./np.sqrt(m) * hadamard(new_dim)[:,:n][indices] * signs.squeeze()
        return lin_op.hmul(RHD.T).T



def gaussian(lin_op, m):
    """
    Gaussian embedding.

    Parameters
    ----------
    lin_op : user-defined linear operator (see LinearOperator class for requirements), with shape (n,d)
    m : sketch size, np.int

    Returns
    -------
    np.ndarray with shape (m,d)

    Notes
    -----
    Samples S an (m,n) Gaussian matrix with i.i.d. entries with zero mean and variance 1/m.
    Performs matrix multiplication lin_op.hmul(S.T)
    """
    lin_op = LinearOperator(lin_op)
    n = lin_op.shape[0]
    St_ = 1./np.sqrt(m) * np.random.randn(n, m)
    return lin_op.hmul(St_).T



def sjlt(lin_op, m, hash_vec=None, sign_vec=None):
    """
    Sparse Johnson-Lindenstrauss Transform (a.k.a. Count Sketch).

    Parameters
    ----------
    lin_op : user-defined linear operator (see LinearOperator class for requirements), with shape (n,d)
    m : sketch size, np.int

    Returns
    -------
    np.ndarray with shape (m,d)

    Notes
    -----
    Checks whether lin_op has attribute 'row slice'. 
    If not, it samples S an (m,n) SJLT (one non-zero entry per column sampled uniformly at random) 
    and then performs lin_op.hmul(S.T) to compute the sketched matrix. 
    """
    
    lin_op = LinearOperator(lin_op)
    n, d = lin_op.shape
    if hash_vec is None:
        hash_vec = np.random.choice(m, n, replace=True)
    if sign_vec is None:
        sign_vec = np.random.choice([-1,1], n, replace=True)

    if hasattr(lin_op, 'row_slice'):
        SX = np.zeros((m, d))
        for j in range(n):
            h = hash_vec[j]
            g = sign_vec[j]
            SX[h, :] += g * lin_op.row_slice([j]).squeeze()
    else:
        St_ = np.zeros((n, m))
        St_[range(n), hash_vec] = sign_vec[range(n)]
        SX = lin_op.hmul(St_).T
    return SX



SKETCH_FN = {'srht': srht, 'gaussian': gaussian, 'sjlt': sjlt}






def test_sjlt():
    n, d, m = 8, 2, 4
    hash_vec = np.array([0, 0, 1, 3, 2, 1, 0, 1])
    sign_vec = np.array([1, -1, 1, 1, 1, 1, -1, -1])
    S = np.zeros((m,n))
    S[hash_vec, range(n)] = sign_vec[range(n)]

    A = np.random.randn(n,d)
    lin_op = LinearOperator(A)
    lin_op_2 = LinearOperator()
    lin_op_2.mul = lin_op.mul; lin_op_2.hmul = lin_op.hmul; lin_op_2.shape = lin_op.shape
    
    assert hasattr(lin_op, 'row_slice') == True
    assert hasattr(lin_op_2, 'row_slice') == False
    
    SA = S @ A 
    SAlinop = sjlt(lin_op, m, hash_vec=hash_vec, sign_vec=sign_vec)
    SAlinop2 = sjlt(lin_op_2, m, hash_vec=hash_vec, sign_vec=sign_vec)

    assert np.linalg.norm(SA - SAlinop) < 1e-8
    assert np.linalg.norm(SA - SAlinop2) < 1e-8

    print('sjlt test successfully passed')



def test_srht(n=10, d=2, m=4):
    
    new_dim = 2**(np.int(np.ceil(np.log(n) / np.log(2))))
    assert n <= new_dim and new_dim < 2*n

    indices = np.sort(np.random.choice(np.arange(new_dim), m, replace=False))
    signs = np.random.choice([-1,1], n, replace=True).reshape((-1,1))

    A = np.random.randn(n,d)
    A, _, _ = np.linalg.svd(A, full_matrices=False)

    lin_op = LinearOperator(A)
    lin_op_2 = LinearOperator()
    lin_op_2.mul = lin_op.mul; lin_op_2.hmul = lin_op.hmul; lin_op_2.shape = lin_op.shape
    
    assert hasattr(lin_op, 'row_slice') == True
    assert hasattr(lin_op_2, 'row_slice') == False
    
    SAlinop = srht(lin_op, m, indices=indices, signs=signs)
    SAlinop2 = srht(lin_op_2, m, indices=indices, signs=signs)

    assert np.linalg.norm(SAlinop - SAlinop2) < 1e-8

    SAlinop = srht(lin_op, new_dim)
    assert np.linalg.norm(SAlinop.T @ SAlinop - np.eye(d)) < 1e-8
    SAlinop2 = srht(lin_op_2, new_dim)
    assert np.linalg.norm(SAlinop2.T @ SAlinop2 - np.eye(d)) < 1e-8

    print('srht test successfully passed')