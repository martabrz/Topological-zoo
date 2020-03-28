import numpy as np
import scipy.linalg as la

def chop(M, tol = 1e-15):
    """ Removes values which are numerical zeroes. Similar to Chop function in Mathematica. """
    if np.isscalar(M):
        if np.iscomplex(M):
            if M.real < tol:
                M = 1j*M.imag
            elif M.imag < tol:
                M = M.real
            elif M.real < tol and M.imag < tol:
                M = 0.0
        else:
            if M < tol:
                M = 0.0
    else:
        if np.iscomplexobj(M):
            M.real[np.abs(M.real) < tol] = 0.0
            M.imag[np.abs(M.imag) < tol] = 0.0
        else:
            M[np.abs(M) < tol] = 0.0
    return M

def eig_(M, sortby = 'real'):
    """ Returns the eigenvalues, eigenvectors (left and right if the matrix is
    non-symmetric) of a matrix M in an ascending order. sortby = 'real', 'imag' or 'abs'. """
    if np.allclose(M, M.T.conj()):
        evals, evecs = la.eigh(M)
        return evals, evecs
    else:
        evals, evecs_l, evecs_r = la.eig(M, left = True, right = True)
        if sortby == 'real':
           sort_p = (evals.real).argsort()
        elif sortby == 'imag':
           sort_p = (evals.imag).argsort()
        elif sortby == 'abs':
           sort_p = (np.abs(evals)).argsort()
        evals = evals[sort_p]
        evecs_r = evecs_r[:, sort_p]
        evecs_l = evecs_l[:, sort_p]
        return evals, evecs_l, evecs_r
