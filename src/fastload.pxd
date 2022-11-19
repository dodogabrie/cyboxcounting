from libc.stdio cimport *
from cython.parallel import prange
from libc.stdlib cimport atof, free, malloc
from libc.string cimport strtok, strncmp, strlen, strcpy
import numpy as np
cimport cython, numpy as np # Make numpy work with cython

# C definitions ###############################################
ctypedef np.double_t DTYPE_t
cdef extern from "stdio.h":
    FILE *fopen(const char *, const char *)
    int fclose(FILE *)
    ssize_t getline(char **, size_t *, FILE *)
############################################################### 
cdef inline int StartsWith(const char *a, const char *b):
    if strncmp(a, b, strlen(b)) == 0: return 1
    return 0;


cdef (int, int) get_dimension(char * , char * , char * )
cdef int get_data(double * , FILE * , char *, int , char * , char * )
