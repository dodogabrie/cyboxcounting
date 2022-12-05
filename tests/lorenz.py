import sys
import os
sys.path.append('../src/')
sys.path.append('../build/')
import numpy as np
import time
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from boxcounting import boxcounting
from compute_dim import fit_show, save_output, dim_df
from scipy.optimize import curve_fit
from scipy.integrate import solve_ivp

# Lorenz paramters and initial conditions.
sigma, beta, rho = 10, 2.667, 28
u0, v0, w0 = 0, 1, 1.05

def lorenz(t, X, sigma, beta, rho):
    """The Lorenz equations."""
    u, v, w = X
    up = - sigma * (u - v)
    vp = rho * u - v - u * w
    wp = - beta*w + u * v
    return up, vp, wp

def solver(T, n, x0):
    # Integrate the Lorenz equations.
    def Jac(t, X, sigma, beta, rho):
        return [[-sigma, sigma, 0], [rho, -1, -X[0]], [X[1], X[0], - beta]]
    t = np.linspace(0, T, n)
    soln = solve_ivp(lorenz, (0, T), x0, args=(sigma, beta, rho), 
                     t_eval=t, dense_output=False,)# method="Radau", jac=Jac)
    # Interpolate solution onto the time grid, t.
    x, y, z = soln.y
    return x, y, z

def write_data(T, n, data_folder, data_file, n_files, hot_start = 0):
    x0 = (u0, v0, w0)
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
        x, y, z = solver(T, n, x0)
        np.savetxt(data_folder + data_file + f'_{i + n_prev_files}.txt',
                   np.column_stack((x, y, z)))

def compute_dimension(data_file, max_level, min_level = 1, multi = False, size = 28, num_tree = 1, output = None):
    bc = boxcounting()
    folderfiles = os.listdir(data_file)
    bc.set_data_file(data_file + folderfiles[0])
    bc.initialize(max_level, size = size, num_tree = num_tree)
    bc.fill_tree()
    if len(folderfiles)>1:
        for i, file in enumerate(folderfiles[1:], 1):
            print(i, end = '\r')
            bc.set_data_file(data_file + file)
            bc.fill_tree()
        bc.count_occupation()
    if output is not None: save_output(bc, output, data_file)
    fit_show(bc, min_index = min_level)

def plot3d(data_file):
    # Plot
    fig = go.Figure()
    n = 0
    for file in os.listdir(data_file):
        data = np.loadtxt(data_file + file)
        data = data[::10]
        n += len(data)
        x, y, z = data.T
        fig.add_trace(go.Scatter3d(x=x, y=y, z=z, mode='markers', marker=dict(size = 1)))
       # fig = go.Figure(data=[go.Scatter3d(x=x, y=y, z=z,
       #                                mode='markers', marker=dict(size = 1))])
#    fig.write_html("data/lorenz.html")
    print("Plotting", n, "points")
    fig.show()
    return 

if __name__ == "__main__":
    data_folder = "data/lorenz_multi_short/"
    output = "data/output/lorenz.txt"
    data_file = "lorenz_short"
    T = 1000
    n = int(1e7)
    max_level = 11
    min_level = 3
    num_files = 6
#    write_data(T, n, data_folder, data_file, num_files, hot_start = 1)
#    plot3d(data_folder)
    compute_dimension(data_folder, max_level, min_level = min_level, multi = True, num_tree=2, output=output)
    dim_df(file = output, block_dim=3)    
