import sys
import os
sys.path.append('../src/')
sys.path.append('../build/')
import numpy as np
import time
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
#    x0 = np.array(x0).astype(np.double)
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
    data0 = np.array((x_init, y_init))
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
        next_x = x[-1]
        np.savetxt(data_folder + data_file + f'_{i + n_prev_files}.txt', x)
        data0 = next_x

def compute_dimension(data_file, max_level, min_level = 1, multi = False, num_tree = 1, size = 1.3):
    bc = boxcounting()
    if multi:
        folderfiles = os.listdir(data_file)
        bc.set_data_file(data_file + folderfiles[0])
        bc.initialize(max_level, size = size, num_tree = num_tree)
        bc.fill_tree()
        for i, file in enumerate(folderfiles[1:], 1):
            print(i, end = '\r')
            bc.set_data_file(data_file + file)
            bc.fill_tree()
        bc.count_occupation()
    else:
        bc.set_data_file("data/"+data_file+".txt")
        bc.initialize(max_level)
        bc.fill_tree()
        bc.count_occupation()
#    fit_show(bc, min_index = min_level)
    return bc

def plot_map(data_folder, data_file):
#    fig = plt.figure()
    for datai in os.listdir(data_folder):
        datai = np.loadtxt(data_folder + datai)
#        plt.scatter(data[:, 0], data[:, 1])
#    plt.show()
    return 

def get_nodes(data_file, max_level, min_level = 1, multi = False):
    list_x = []
    list_y = []
    list_r = []
    list_big_x = []
    list_big_y = []
    list_big_r = []
    for max_level in range(1, 8):
        xx = []
        yy = []
        rr = []
        bc = compute_dimension(data_file, max_level, min_level, multi)
        output = bc.nodes 
        nodes = output[0]
        n = output[1]

        print(f"Found array of {n} nodes")
#        print(nodes)
        big_r = 0
        x_r   = 0
        y_r   = 0
        for data in nodes:
            r = data[2]
            x = data[0] 
            y = data[1] 
            if big_r < r: 
                big_r = r 
                x_r = x 
                y_r = y
            xx.append(x)
            yy.append(y)
            rr.append(r)
        list_x.append(xx)
        list_y.append(yy)
        list_r.append(rr)
        list_big_x.append(x_r)
        list_big_y.append(y_r)
        list_big_r.append(big_r)
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    i = 1
    for xx, yy, rr, x_r, y_r, big_r in zip(list_x, list_y, list_r, list_big_x, list_big_x, list_big_r):
        fig = plt.figure(figsize=(6, 6))
        for datai in os.listdir(data_file):
            datai = np.loadtxt(data_file + datai)
            plt.scatter(datai[:, 0], datai[:, 1], s = 1, alpha = 0.1, c = 'tab:blue')

        for x, y, r in zip(xx, yy, rr):
            plt.gcf().gca().add_patch(patches.Rectangle((x-r, y-r), 2*r, 2*r, fill=False, lw = 0.4))
            plt.scatter(x, y, alpha=0)


        plt.xlim(x_r - big_r, x_r + big_r)
        plt.ylim(y_r - big_r, y_r + big_r)
        print(f"writing {i}")
        plt.savefig(f"figures/henon_map_with_tree/Tree_{i}.png", dpi = 200)
        i+=1
#        plt.show()
    return 


if __name__ == "__main__":
    data_folder = "data/henon_debug/"
    data_file = "henon"
    n = int(1e4)
    max_level = 3
    min_level = 1
    num_files = 3
    write_data(n, data_folder, data_file, num_files, hot_start = 0)
#    plot_map(data_folder, "henon_1.txt")
#    compute_dimension(data_folder, max_level, min_level = min_level, multi = True)
    get_nodes(data_folder, max_level, min_level = min_level, multi = True)
 
