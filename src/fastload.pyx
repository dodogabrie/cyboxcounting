""" This Module read a 2 COLUMNS file of data in a fast way """

# Main function ###############################################
def fastload(filename, comments = '#', delimiter = ' ', usecols = None):
    """
    Function for fast extraction of data from txt file.
    NOTE: do not import matplotlib before importing this module, in that 
    case the module will not work...
    Parameters
    ----------
    filename : 'b'string
        String containing the file .txt and his path preceeded by the letter b.
        For example b'mydata.txt'
    Return: 2d numpy array
        Array of x/y containing the data in the txt columns.
    """
    cdef int i
    #cdef int len_comment = len(comments), len_delimiter = len(delimiter), len_fname = len(filename)

    cdef char * cdelimiter = <char*>malloc(10 * sizeof(char))
    cdef char * ccomments  = <char*>malloc(10 * sizeof(char))
    cdef char * fname      = <char*>malloc(100 * sizeof(char))
    strcpy(cdelimiter, delimiter.encode('utf-8'))
    strcpy(ccomments, comments.encode('utf-8'))
    strcpy(fname, filename.encode('utf-8'))

    Ndata, Ncol = get_dimension(fname, cdelimiter, ccomments)

    x = take_data(data, fname, Ndata, Ncol, cdelimiter, ccomments, cols) 
    return x
###############################################################

cdef (int, int) get_dimension(char * fname, char * delimiter, char * comments):
    cfile = fopen(fname, "rb")
    if cfile == NULL:
        raise FileNotFoundError(2, "No such file or directory: '%s'" % fname)
    cdef int Ndata=0, Ncol=0
    cdef char * line = NULL
    cdef size_t l = 0
    cdef ssize_t read
    while True:
        read = getline(&line, &l, cfile)
        if read == -1: break
        if Ndata == 0:
            line = strtok(line, delimiter)
            if StartsWith(line, comments): continue
            else:
                while line != NULL:
                    line = strtok(NULL, delimiter)
                    Ncol += 1
        Ndata += 1
    free(line)
    fclose(cfile)
    return Ndata, Ncol

# Core function (do the hard word) ############################
cdef void take_data(DTYPE_t[:,:] data, char * fname, int Ndata, int Ncol, 
        char * delimiter, char * comments, np.int_t[:] cols):
    cfile = fopen(fname, "rb")
    cdef int i, j = 0, j_max = 0
    cdef int col_counter = 0 # column index
    cdef int c = cols[col_counter]
    cdef char * line = NULL
    cdef char * token 
    cdef size_t l = 0
    cdef ssize_t read = 0
    for i in range(Ndata):
        read = getline(&line, &l, cfile)
        if read == -1: break
        token = strtok(line, delimiter)
        if StartsWith(line, comments): continue
        for j in range(Ncol):
            if j == c:
                data[i][col_counter] = atof(token)
                col_counter += 1
                c = cols[col_counter]
            token = strtok(NULL, delimiter)
        col_counter = 0
        c = cols[col_counter]
    free(line)
    fclose(cfile)
###############################################################

def get_data_shape(filename, comments = '#', delimiter = ' '):
    """
    Function for fast extraction of data from txt file.
    NOTE: do not import matplotlib before importing this module, in that 
    case the module will not work...
    Parameters
    ----------
    filename : 'b'string
        String containing the file .txt and his path preceeded by the letter b.
        For example b'mydata.txt'
    Return: 2d numpy array
        Array of x/y containing the data in the txt columns.
    """
    def turn_utf8(data):
        "Returns an utf-8 object on success, or None on failure"
        try: # Try to encode the data as utf-8
            return data.encode('utf-8')
        except AttributeError: # if already utf-8
            return data

    cdef bytes cdelimiter = turn_utf8(delimiter)
    cdef bytes ccomments = turn_utf8(comments)
    cdef bytes fname = turn_utf8(filename)

    Ndata, Ncol = get_dimension(fname, cdelimiter, ccomments)
    return Ndata, Ncol
#
