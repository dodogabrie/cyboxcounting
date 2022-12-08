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
    cdef tree_t * tree
    cdef int initialized
    cdef int * _occ
    cdef double * _eps
    cdef int max_level, n, dim, tot_data, num_tree, actual_tree
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

    def set_data_file(self, filename, comments = '#', delimiter = ' '):
        strcpy(self.cdelimiter, delimiter.encode('utf-8'))
        strcpy(self.ccomments, comments.encode('utf-8'))
        strcpy(self.fname, filename.encode('utf-8'))

    def initialize(self, int max_level, int num_tree = 1, double size = 0):
        cdef int Ncol, Ndata, i
        self.max_level = max_level
        self.num_tree = num_tree
        self.tree = <tree_t*>malloc(num_tree*sizeof(tree_t))
        Ndata, Ncol = get_dimension(self.fname, self.cdelimiter, self.ccomments)
        for i in range(num_tree):
            self.tree[i] = create_tree(Ncol, 0)
        self.initialized = 1
        self.dim = Ncol
        self.initialize_size(size)

    def fill_tree(self):
        cdef int eof = 0, n = 0
        cdef double * x
        x = <double*>malloc(self.dim * sizeof(double))
        self.cfile = fopen(self.fname, "rb")
        eof = get_data(x, self.cfile, self.token, self.dim, self.cdelimiter, self.ccomments) 
        while eof != -1:
            for i in range(self.num_tree):
                recursive_occupation(&self.tree[i], x, self.max_level, self.dim)
            eof = get_data(x, self.cfile, self.token, self.dim, self.cdelimiter, self.ccomments) 
            n = n + 1
        fclose(self.cfile)
        self.n = n
        self.tot_data += n
        free(x)

    def count_occupation(self):
        cdef int i
        for i in range(self.num_tree):
            recursive_count(&self.tree[i], &self._occ[i*self.max_level], self.max_level, self.dim)

    def IO_initialize(self):
        self.cdelimiter = <char*>malloc(10 * sizeof(char))
        self.ccomments  = <char*>malloc(10 * sizeof(char))
        self.fname      = <char*>malloc(100 * sizeof(char))
        self.token = NULL

    def initialize_size(self, double size):
        cdef double radi, sizeM = 0
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
        for i in range(self.dim):
            for j in range(self.num_tree):
                self.tree[j].centr[i] = self.cmid[i]
            radi =  0.5*(self.M[i]-self.m[i])
            if sizeM < radi:
                sizeM = radi

        if size != 0:
            sizeM = size
        self.tree[0].radi = sizeM
        for j in range(1, self.num_tree):
            self.tree[j].radi = sizeM + sizeM / (j + 1)
        self.eps0 = sizeM

        self._occ = <int*>calloc(self.max_level*self.num_tree, sizeof(int))
        self._eps = <double*>malloc(self.max_level*self.num_tree*sizeof(double))
        self._eps[0] = self.eps0
        if self.num_tree > 1:
            for j in range(1, self.num_tree):
                self._eps[j*self.max_level] = self.eps0 + self.eps0 / (j + 1)
        for i in range(1, self.max_level): 
            for j in range(self.num_tree):
                self._eps[i + j*self.max_level] = self._eps[i-1 + j*self.max_level]/2
        self.initialized = 1

    @property 
    def occ(self):
        occc = np.ones(self.max_level*self.num_tree).astype(int)
        for i in range(self.max_level*self.num_tree):
            occc[i] = self._occ[i]
        return occc

    @property 
    def eps(self):
        epss = np.zeros(self.max_level*self.num_tree).astype(np.double)
        for i in range(self.max_level*self.num_tree):
            epss[i] = self._eps[i]/self.eps0
        return epss

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
    @property 
    def num_tree(self):
        return self.num_tree
    def free(self):
        cdef int i 
        for i in range(self.num_tree):
            free_tree(&self.tree[i], self.dim)
    @property
    def nodes(self):
        cdef int n 
        for i in range(self.num_tree):
            n = get_num_nodes(self.tree, 0, self.dim)
        nodes = np.zeros((n, self.dim + 1)).astype(np.double)
        get_nodes(self.tree, nodes, self.dim, 0)
        return nodes, n

cdef tree_t create_tree(int n, int level):
    cdef int i
    cdef tree_t tree
    cdef int num_quadrants = 2**n # Quadrants for n-dim problem: 2^n
    tree.child   = <tree_t*>calloc(num_quadrants, sizeof(tree_t))
    tree.centr = <double*>malloc(n*sizeof(double))
    tree.level = level
    return tree

cdef void free_tree(tree_t * tree, int dim):
    cdef int i
    if tree.level != 0:
        for i in range(dim):
            free_tree(&tree.child[i], dim)
    else: return 
    free(tree.centr)
    free(tree.child)
    return 

cdef void recursive_occupation(tree_t * tree, double * x, int max_level, int dim):
    cdef tree_t * next_tree
    if tree.level < max_level:
#        if tree.filled == 0:
#            tree.filled += 1
        next_tree = next_child(tree, x, dim)
        recursive_occupation(next_tree, x, max_level, dim)
    return

cdef void recursive_count(tree_t * tree, int * occ, int max_level, int dim):
    cdef int i
    cdef int num_quadrants = 2**dim
    if tree.level < max_level:
        if tree.radi != 0:
            occ[tree.level] += 1
            for i in range(num_quadrants):
                recursive_count(&tree.child[i], occ, max_level, dim)
    return

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
            if tree.centr[i] + tree.radi < x[i]:
                printf("out of bound")
            inc_bool = 1
            next_mid[i] = tree.centr[i] + next_radi 
        else: 
            inc_bool = 0
            next_mid[i] = tree.centr[i] - next_radi
        quadrant += inc * inc_bool
        inc += inc

    if tree.child[quadrant].radi == 0:
        tree.child[quadrant] = create_tree(dim, tree.level + 1)
        for i in range(dim):
            tree.child[quadrant].centr[i] = next_mid[i]
        tree.child[quadrant].radi = next_radi
    free(next_mid)
    return &tree.child[quadrant]

cdef int get_num_nodes(tree_t * tree, int n_nodes, int dim):
    cdef int i, num_quadrants = 2**dim
    if tree.radi != 0:
        n_nodes += 1
        for i in range(num_quadrants):
            n_nodes = get_num_nodes(&tree.child[i], n_nodes, dim)
    return n_nodes

cdef int get_nodes(tree_t * tree, np.double_t[:,:] x, int dim, int line):
    cdef int i, num_quadrants = 2**dim
    if tree.radi != 0:
        for i in range(dim):
            x[line,i] = tree.centr[i]
        x[line, dim] = tree.radi
        line += 1
        for i in range(num_quadrants):
            line = get_nodes(&tree.child[i], x, dim, line)
    return line


# Creare array con  
# Centro (x, y) 
# raggio (eps)
# plottarlo con plt.gcf().gca().add_patch(patches.Rectangle((n.x0, n.y0), n.width, n.height, fill=False))
