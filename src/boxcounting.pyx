# distutils: language=c++
import numpy as np
cimport cython
cimport numpy as np
cimport tree
from libc.stdlib cimport malloc, calloc, free
from tree cimport tree_t
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

ctypedef np.int_t INT_t
ctypedef np.double_t DOUBLE_t

cdef class boxcounting:
    cdef tree_t tree
    cdef int * occ
    cdef int max_level
    cdef int n_data
    cdef double final_dim
    cdef double final_var
    cdef int dim 

    def __init__(self, int n):
        self.tree = create_tree(n)

    @cython.boundscheck(False)  # Deactivate bounds checking
    @cython.wraparound(False)   # Deactivate negative indexing.
    def occupation(self, DOUBLE_t[:,:] x, int max_level):
        self.max_level = max_level
        cdef np.ndarray[INT_t, ndim=1, mode='c'] occ 
        occ = np.zeros(max_level).astype(int)
        cdef actual_level = 0
        cdef int i, n = x.shape[0], dim = x.shape[1]
        self.dim = dim 
        cdef double mean, mid, radi, radi_Max = 0.
        self.n_data = n

        # Compute min and max for each dim
        for i in range(dim):
            mid = 0.5*(np.max(x[:, i]) + np.min(x[:, i]))
            radi = 0.5*(np.max(x[:, i]) - np.min(x[:, i]))
            if radi > radi_Max: radi_Max = radi
            self.tree.centr[i] = mid
        self.tree.radi = radi_Max

        for i in range(n):
            recursive_occupation(&self.tree, x[i], occ, max_level, actual_level, dim)

        self.occ = <int*>malloc(max_level*sizeof(int))
        for i in range(len(occ)):
            self.occ[i] = occ[i]
        return occ

    def fit_dim(self, int min_index = 1, int max_index = 0):
        if max_index == 0: max_index = self.max_level
        cdef np.ndarray[INT_t, ndim=1, mode='c'] occ 
        cdef int i
        occ = np.empty(self.max_level).astype(int)
        for i in range(self.max_level):
            occ[i] = self.occ[i]

        y = np.log2(occ)[min_index:max_index]
        den = np.arange(min_index, max_index)
        dim = y/den
        def f(x, m, q):
            return m * x + q
    
        init = [1., 0]
        popt, pcov = curve_fit(f, den, y, p0=init)
        x_array = np.linspace(np.min(den), np.max(den), 100)
        D = popt[0]
        var = np.sqrt(np.diag(pcov))[0]
        self.final_dim = D
        self.final_var = var
        return

    def fit_show(self, int min_index = 1, int max_index = 0):
        if max_index == 0: max_index = self.max_level
        cdef np.ndarray[INT_t, ndim=1, mode='c'] occ 
        cdef int i
        occ = np.empty(self.max_level).astype(int)
        for i in range(self.max_level):
            occ[i] = self.occ[i]

        y = np.log2(occ)[min_index:max_index]
        den = np.arange(min_index, max_index)
        dim = y/den
        def f(x, m, q):
            return m * x + q
    
        init = [1., 0]
        popt, pcov = curve_fit(f, den, y, p0=init)
        x_array = np.linspace(np.min(den), np.max(den), 100)
        D = popt[0]
        var = np.sqrt(np.diag(pcov))[0]
        print(f'D = {popt[0]} pm {np.sqrt(np.diag(pcov))[0]}')
        print(f'Last D evaluated: {dim[-1]}')
        self.final_dim = D
        self.final_var = var
        
        # Plot dei risultati
        fig, axs = plt.subplots(1,2,figsize=(15,7))
        plt.suptitle(r'Fattore di scala: $\epsilon = 2^n$', fontsize = 20)
        plt.setp(axs[0].get_xticklabels(), fontsize=13)
        plt.setp(axs[0].get_yticklabels(), fontsize=13)
        plt.setp(axs[1].get_xticklabels(), fontsize=13)
        plt.setp(axs[1].get_yticklabels(), fontsize=13)
        axs[0].plot(x_array, f(x_array, *popt))
        axs[0].scatter(den, y, c='k')
        axs[0].set_xlim(np.min(den)-1,np.max(den)+1)
        axs[0].set_ylim(np.min(y)-1,np.max(y)+1)
        axs[0].grid()
        axs[0].set_xlabel(r'$\log(1/\epsilon)$', fontsize = 20)
        axs[0].set_ylabel(r'$\log(N(\epsilon))$', fontsize = 20)
        axs[0].set_title(fr'$y = mx$ $\rightarrow$ m = {popt[0]:.3f} $\pm$ {np.sqrt(np.diag(pcov))[0]:.3f}', fontsize = 20)
        n_rel = self.n_data/occ[min_index:max_index]
        axs[1].scatter(den, n_rel, c='k')
        axs[1].plot(x_array, np.ones(len(x_array)), linestyle='--', c='r')
        axs[1].set_yscale('log')
        axs[1].grid()
        axs[1].set_xlabel(r'$\log(1/\epsilon)$', fontsize = 20)
        axs[1].set_ylabel(r'$\left<n(\epsilon)\right>$', fontsize = 20)
        axs[1].set_title('# medio dati in un quadrato', fontsize = 20)
        plt.show()
        return
    def free(self):
        free_tree(&self.tree, self.dim)
        return 

cdef inline double mid_val(double x, double y):
    return (x + y) * 0.5

cdef tree_t create_tree(int n):
    cdef tree_t tree
    cdef int num_quadrants = 2**n # Quadrants for n-dim problem: 2^n
    tree.child   = <tree_t*>calloc(num_quadrants, sizeof(tree_t))
    tree.centr = <double*>malloc(n*sizeof(double))
    tree.radi = 0
    tree.filled = 0
    tree.initialized = 1
    return tree

cdef void free_tree(tree_t * tree, int dim):
    if tree.initialized != 0:
        for i in range(dim):
            free_tree(&tree.child[i], dim)
    else: return 
    free(tree.centr)
    free(tree.child)
    return 

@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
cdef void recursive_occupation(tree_t * tree, DOUBLE_t[:] x, INT_t[:] occ, int max_level, int level, int dim):
    cdef int next_quadrant
    cdef tree_t * next_tree
    cdef int i
    if level < max_level:
        if tree.filled == 0:
            tree.filled += 1
            occ[level] += 1
        level += 1
        next_tree = next_child(tree, x, dim)
        recursive_occupation(next_tree, x, occ, max_level, level, dim)
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
        tree.child[quadrant] = create_tree(dim)
        for i in range(dim):
            tree.child[quadrant].centr[i] = next_mid[i]
            tree.child[quadrant].radi = next_radi
    free(next_mid)
    return &tree.child[quadrant]
