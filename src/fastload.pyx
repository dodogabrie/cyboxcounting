""" This Module read a 2 COLUMNS file of data in a fast way """

cdef (int, int) get_dimension(char * fname, char * delimiter, char * comments):
    cfile = fopen(fname, "rb")
    if cfile == NULL:
        raise FileNotFoundError(2, "No such file or directory: '%s'" % fname)
    cdef int Ndata=-1, Ncol=0
    cdef char * line = NULL
    cdef size_t l = 0
    cdef int read = 0
    while read != -1:
        read = getline(&line, &l, cfile)
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
cdef int get_data(double * data, FILE * cfile, char * token, int Ncol, char * delimiter, char * comments):
    cdef int j = 0
    cdef char * line = NULL
    cdef size_t l = 0
    cdef ssize_t read = 0
    cdef int skip_comments = 0
    while skip_comments == 0:
        read = getline(&line, &l, cfile)
        if read == -1: break
        token = strtok(line, delimiter)
        if StartsWith(line, comments): continue
        for j in range(Ncol):
            data[j] = atof(token)
            token = strtok(NULL, delimiter)
        skip_comments = 1
    free(line)
    free(token)
    return read
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
