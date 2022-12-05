import sys
import os
sys.path.append('../src/')
sys.path.append('../build/')
import numpy as np
import time
import matplotlib.pyplot as plt
from numba import njit
import plotly.graph_objects as go
from boxcounting import boxcounting
from compute_dim import fit_show
from scipy.optimize import curve_fit
from scipy.integrate import solve_ivp

a = 1.4
b = 0.3
x_init = 1
y_init = 1


@njit
def solver(n, x0):
    # Integrate the Lorenz equations.
    x0 = np.array(x0).astype(np.double)
    # termalization
    for i in range(50):
        xs0 = x0[0]
        xs1 = x0[1]
        x0[0] = 1 - a * xs0*xs0 + xs1
        x0[1] = b * xs0
    # Solver
    x = np.empty((n, 2))
    x[0] = x0
    for i in range(n-1):
        x[i+1, 0] = 1 - a * x[i, 0]*x[i, 0] + x[i, 1] 
        x[i+1, 1] = b * x[i, 0]
    return x

def write_data(n, data_folder, data_file, n_files, hot_start = 0):
    data0 = (x_init, y_init)
    if not os.path.exists(data_folder):
        os.mkdir(data_folder)

    n_prev_files = 0
    files = os.listdir(data_folder)
    if hot_start and files:
        n_prev_files = len(files)
        for f in files:
            if f.endswith(f"{n_prev_files-1}.txt"): 
                x0 = np.loadtxt(data_folder + f)[-1]
    for i in range(n_files):
        print(i + n_prev_files, end = "\r")
        x = solver(n, data0)
        np.savetxt(data_folder + data_file + f'_{i + n_prev_files}.txt', x)

def compute_dimension(data_file, max_level, min_level = 1, multi = False):
    bc = boxcounting()
    if multi:
        folderfiles = os.listdir(data_file)
        bc.set_data_file(data_file + folderfiles[0])
        bc.initialize(max_level, size = 1.3, num_tree = 3)
        bc.fill_tree()
        for i, file in enumerate(folderfiles[1:], 1):
            print(i, end = '\r')
            bc.set_data_file(data_file + file)
            bc.fill_tree()
        bc.count_occupation()
        print(bc.tot_data)
    else:
        bc.set_data_file("data/"+data_file+".txt")
        bc.initialize(max_level)
        bc.fill_tree()
        bc.count_occupation()
    fit_show(bc, min_index = min_level)

def plot_map(data_folder, data_file):
    data = np.loadtxt(data_folder + data_file)
    print(data.shape)
    print(data[:, 0].shape)
    plt.scatter(data[:, 0], data[:, 1])
    plt.show()
    return 


if __name__ == "__main__":
    data_folder = "data/henon_debug/"
    data_file = "henon"
    n = int(1e5)
    max_level = 12
    min_level = 1
    num_files = 2
    write_data(n, data_folder, data_file, num_files, hot_start = 0)
#    plot_map(data_folder, "henon_1.txt")
    compute_dimension(data_folder, max_level, min_level = min_level, multi = True)
 
