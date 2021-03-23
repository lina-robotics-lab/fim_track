import cvxpy as cp
import numpy as np
import networkx as nx
def consensus_weight(A):
    '''
        Given a sparsity A, find the best doubly stochastic consensus weight matrix W that
        matches the sparsity pattern A and minimizes ||W-1/n 11^T||_2.
    '''
    M = 2 # The 'BIG M' parameter. It is big compared to the entries of W.
    n_dim = len(A)

    W = cp.Variable(shape = (n_dim,n_dim),symmetric=True)

    constraints = [W>=-M*A,W<=M*A,W>=0,cp.sum(W,axis=1)==1]
    obj = cp.Minimize(cp.norm(W-1/n_dim * np.ones((n_dim,n_dim))))
    prob = cp.Problem(obj,constraints)
    prob.solve()

    W.value[W.value<1e-4]=0
    return W.value