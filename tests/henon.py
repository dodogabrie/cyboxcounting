import sys
import os
sys.path.append('../src/')
sys.path.append('../build/')
import numpy as np
import time
from numba import njit
import plotly.graph_objects as go
from boxcounting import boxcounting
from compute_dim import fit_show, create_tree
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

def get_nodes(data_file, max_level, min_level = 1, multi = False):
    list_x = []
    list_y = []
    list_r = []
    list_big_x = []
    list_big_y = []
    list_big_r = []
    max_level_list = [1, 2, 3, 7]
    bc = create_tree(data_file, max_level, min_level, size = 1.3)
    nodes, levels, n = bc.nodes 
    for max_level in max_level_list:
        xx = []
        yy = []
        rr = []
        big_r = 0
        x_r   = 0
        y_r   = 0
        for data, lev in zip(nodes, levels):
            if lev < max_level:
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
    plt.rc('font', **{'size':15})
    fig, axs = plt.subplots(nrows = 1, ncols = len(max_level_list), figsize=(12, 3), sharey=True )
    n_files = len(os.listdir(data_file))
    for ax, xx, yy, rr, x_r, y_r, big_r in zip(axs, list_x, list_y, list_r, list_big_x, list_big_x, list_big_r):
        i = 1
        for datai in os.listdir(data_file):
            datai = np.loadtxt(data_file + datai)
            red_dot = datai[-1, :]
            ax.scatter(datai[:, 0], datai[:, 1], s = 1, alpha = 0.1, c = 'tab:blue')
            if i == n_files:
                ax.scatter(red_dot[0], red_dot[1], s = 40, alpha = 1, c = 'tab:red', edgecolor = 'k')
            i += 1

        for x, y, r in zip(xx, yy, rr):
            ax.add_patch(patches.Rectangle((x-r, y-r), 2*r, 2*r, fill=False, lw = 1))
            ax.scatter(x, y, alpha=0)


        ax.set_xlim(x_r - big_r, x_r + big_r)
        ax.set_ylim(y_r - big_r, y_r + big_r)
        print(f"writing {i}")
        i+=1
    plt.savefig(f"figures/henon_map_with_tree/Tree1.png", dpi = 200)
    plt.show()
    return 
def compute_D(datafile, max_level, min_level=1, size = 1.4, num_tree = 1):
    start = time.time()
    bc = create_tree(datafile, max_level, num_tree=num_tree, size = size)
    print("elapsed: ", time.time()-start, "s")
    _, _, fig, ax = fit_show(bc, min_index=min_level)
    import matplotlib.pyplot as plt
#    plt.savefig("figures/henon_dim/bc_dim_henon.png", dpi = 200)
    return


if __name__ == "__main__":
    data_folder = "data/henon_debug/"
    data_file = "henon"
    n = int(1e6)
    max_level = 12
    min_level = 2
    num_tree = 3
    num_files = 3
#    write_data(n, data_folder, data_file, num_files, hot_start = 0)
#    plot_map(data_folder, "henon_1.txt")
    compute_D(data_folder, max_level, min_level=min_level, num_tree=num_tree)
#    get_nodes(data_folder, max_level, min_level = min_level, multi = True)
 
