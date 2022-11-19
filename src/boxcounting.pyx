# distutils: language=c++
import numpy as np
cimport cython
cimport numpy as np
cimport tree
cimport fastload
from fastload cimport get_dimension, get_data
from libc.stdio cimport *
from libc.stdlib cimport malloc, calloc, free
from libc.string cimport strtok, strncmp, strlen, strcpy
from tree cimport tree_t

ctypedef np.int_t INT_t
ctypedef np.double_t DOUBLE_t

cdef class boxcounting:
    cdef tree_t tree
    cdef int initialized
    cdef int * occ
    cdef double * eps
    cdef int max_level, n, dim, tot_data, num_tree
    cdef double final_dim, final_var, eps0
    cdef char * cdelimiter 
    cdef char * ccomments
    cdef char * fname
    cdef char * token
    cdef double * m 
    cdef double * M 
    cdef double * cmid 


    cdef FILE * cfile 

    def __init__(self):
        self.initialized = 0 
        self.tot_data = 0
        self.eps0 = 0
        self.IO_initialize()

    def occupation(self, filename, int max_level, comments = '#', delimiter = ' ', int num_tree = 1):
        cdef int i
        cdef double * x
        cdef double radi
        cdef int Ndata, Ncol
        self.max_level = max_level
        self.num_tree = num_tree
        strcpy(self.cdelimiter, delimiter.encode('utf-8'))
        strcpy(self.ccomments, comments.encode('utf-8'))
        strcpy(self.fname, filename.encode('utf-8'))
    
        Ndata, Ncol = get_dimension(self.fname, self.cdelimiter, self.ccomments)
        self.tree = create_tree(Ncol, 0)
        self.initialized = 1

        x = <double*>malloc(Ncol * sizeof(double))
        self.n = Ndata 
        self.dim = Ncol

        self.initialize_size()
        self.cfile = fopen(self.fname, "rb")
        eof = 0
        while eof != -1:
            eof = get_data(x, self.cfile, self.token, Ncol, self.cdelimiter, self.ccomments) 
            recursive_occupation(&self.tree, x, max_level, self.dim)
        fclose(self.cfile)
        recursive_count(&self.tree, self.occ, max_level, self.dim)
        self.free()

        self.tot_data += self.n

    def IO_initialize(self):
        self.cdelimiter = <char*>malloc(10 * sizeof(char))
        self.ccomments  = <char*>malloc(10 * sizeof(char))
        self.fname      = <char*>malloc(100 * sizeof(char))
        self.token = NULL

    def initialize_size(self):
        cdef double radi, size
        cdef double * x 
        cdef int i, eof = 0
        x = <double*>malloc(self.dim*sizeof(double))

        if self.tot_data == 0:
            self.M = <double*>malloc(self.dim*sizeof(double))
            self.m = <double*>malloc(self.dim*sizeof(double))
            self.cmid = <double*>malloc(self.dim*sizeof(double))

            for i in range(self.dim):
                self.M[i] = -1e8
                self.m[i] = 1e8

        self.cfile = fopen(self.fname, "rb")
        while eof != -1:
            eof = get_data(x, self.cfile, self.token, self.dim, self.cdelimiter, self.ccomments) 
            for i in range(self.dim):
               if self.m[i] > x[i]:
                    self.m[i] = x[i]
               if self.M[i] < x[i]:
                        self.M[i] = x[i]
        fclose(self.cfile)

        for i in range(self.dim):
            self.cmid[i] = 0.5*(self.M[i]+self.m[i])
        size = 0
        for i in range(self.dim):
            self.tree.centr[i] = self.cmid[i]
            radi =  0.5*(self.M[i]-self.m[i])
            if size < radi:
                size = radi
        self.tree.radi = size 
        self.eps0 = size
        self.occ = <int*>malloc(self.max_level*self.num_tree*sizeof(int))
        self.eps = <double*>malloc(self.max_level*self.num_tree*sizeof(double))
        self.eps[0] = self.eps0
        for i in range(self.max_level): 
            self.occ[i] = 0
            self.eps[i] = 0
        self.initialized = 1

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
    def eps0(self):
        return self.eps0
    @property 
    def n(self):
        return self.n
    @property 
    def tot_data(self):
        return self.tot_data
    @property 
    def max_level(self):
        return self.max_level

    def free(self):
        free_tree(&self.tree, self.dim)

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
cdef void recursive_occupation(tree_t * tree, double * x, int max_level, int dim):
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
cdef tree_t * next_child(tree_t * tree, double * x, int dim):
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
    cdef double next_radi = tree.radi*0.5
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
