# distutils: language=c++
import numpy as np
cimport cython
cimport numpy as np
cimport tree
from libc.stdlib cimport malloc, calloc, free
from tree cimport tree_t

ctypedef np.int_t INT_t
ctypedef np.double_t DOUBLE_t

cdef class boxcounting:
    cdef tree_t tree
    cdef int * occ
    cdef int max_level, n_data, dim
    cdef double final_dim, final_var, eps

    def __init__(self, int n):
        self.tree = create_tree(n, 0)

    @cython.boundscheck(False)  # Deactivate bounds checking
    @cython.wraparound(False)   # Deactivate negative indexing.
    def occupation(self, DOUBLE_t[:,:] x, int max_level):
        self.max_level = max_level
        cdef int i, n = x.shape[0], dim = x.shape[1]
        self.dim = dim 
        cdef double mid, radi, radi_Max = 0.
        if self.n_data == 0: self.n_data = n
        else: self.n_data += n

        # Compute min and max for each dim
        for i in range(dim):
            mid = 0.5*(np.max(x[:, i]) + np.min(x[:, i]))
            radi = 0.5*(np.max(x[:, i]) - np.min(x[:, i]))
            if radi > radi_Max: radi_Max = radi
            self.tree.centr[i] = mid
        self.tree.radi = radi_Max
        self.eps = radi_Max

        self.occ = <int*>malloc(max_level*sizeof(int))
        for i in range(max_level): self.occ[i] = 0
        for i in range(n): recursive_occupation(&self.tree, x[i], max_level, dim)
        recursive_count(&self.tree, self.occ, max_level, dim)

    @property 
    def occ(self):
        cdef np.ndarray[INT_t, ndim=1, mode='c'] occ 
        occ = np.zeros(self.max_level).astype(int)
        for i in range(self.max_level):
            occ[i] = self.occ[i]
        return occ
    @property
    def max_level(self):
        return self.max_level
    @property
    def eps(self):
        return self.eps
    @property 
    def n_data(self):
        return self.n_data
    @property 
    def max_level(self):
        return self.max_level

    def free(self):
        free_tree(&self.tree, self.dim)
        return 

cdef tree_t create_tree(int n, int level):
    cdef tree_t tree
    cdef int num_quadrants = 2**n # Quadrants for n-dim problem: 2^n
    tree.child   = <tree_t*>calloc(num_quadrants, sizeof(tree_t))
    tree.centr = <double*>malloc(n*sizeof(double))
    tree.radi = 0
    tree.filled = 0
    tree.initialized = 1
    tree.level = level
    return tree

cdef void free_tree(tree_t * tree, int dim):
    cdef int i
    if tree.initialized != 0:
        for i in range(dim):
            free_tree(&tree.child[i], dim)
    else: return 
    free(tree.centr)
    free(tree.child)
    return 

@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
cdef void recursive_occupation(tree_t * tree, DOUBLE_t[:] x, int max_level, int dim):
    cdef tree_t * next_tree
    if tree.level < max_level:
        if tree.filled == 0:
            tree.filled += 1
        next_tree = next_child(tree, x, dim)
        recursive_occupation(next_tree, x, max_level, dim)
    return

@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
cdef void recursive_count(tree_t * tree, int * occ, int max_level, int dim):
    cdef int i
    cdef int num_quadrants = 2**dim
    if tree.level < max_level:
        if tree.filled != 0:
            occ[tree.level] += 1
            for i in range(num_quadrants):
                recursive_count(&tree.child[i], occ, max_level, dim)
    return

@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
cdef tree_t * next_child(tree_t * tree, DOUBLE_t[:] x, int dim):
    """
    Function that select the quadrant for the next step based on the input data.
    It also define the next node max, min, mid

    Parameters
    ----------
    tree : tree_t
        Tree structure in input at this level.
    x : memoryview of numpy 1d array
        Input data.
    dim: int
        Dimension of x (x.shape[1]).
    
    Returns
    -------
    int
        The integer value correspondind to the next quadrant to go for the 
        data x.
    """
    cdef int i, quadrant = 0 # output quadrant (starting from zero)
    cdef int inc_bool # for each data dimension inc_bool is 0 if the data 
    #                 # is under the mid, otherwise is 1
    cdef int inc = 1  # inc is incremented by himself after each step of the
    #                 # next loop.
    cdef double * next_mid =  <double*>malloc(dim*sizeof(double))
    cdef double next_radi = tree.radi/2.
    for i in range(dim):
        if tree.centr[i] < x[i]: 
            inc_bool = 1
            next_mid[i] = tree.centr[i] + next_radi 
        else: 
            inc_bool = 0
            next_mid[i] = tree.centr[i] - next_radi
        quadrant += inc * inc_bool
        inc += inc

    if tree.child[quadrant].initialized == 0:
        tree.child[quadrant] = create_tree(dim, tree.level + 1)
        for i in range(dim):
            tree.child[quadrant].centr[i] = next_mid[i]
            tree.child[quadrant].radi = next_radi
    free(next_mid)
    return &tree.child[quadrant]
