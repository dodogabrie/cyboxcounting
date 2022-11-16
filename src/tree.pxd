import numpy as np
cimport numpy as np

cdef struct tree_t:
    int filled, level
    int initialized
    double * centr
    double radi
    tree_t * child
