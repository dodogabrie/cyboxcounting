import sys
sys.path.append('../src/')
sys.path.append('../build/')
import numpy as np
import time
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from boxcounting import boxcounting
from compute_dim import fit_show
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

def solver(tmax, n):
    # Integrate the Lorenz equations.
    t = np.linspace(0, tmax, n)
    soln = solve_ivp(lorenz, (0, tmax), (u0, v0, w0), args=(sigma, beta, rho), t_eval=t,
                     dense_output=False)
    # Interpolate solution onto the time grid, t.
    x, y, z = soln.y
    return x, y, z

def write_data(tmax, n, data_file):
    x, y, z = solver(tmax, n)
    np.savetxt("data/" + data_file, np.column_stack((x, y, z)))

def compute_dimension(data_file, max_level, min_level = 1):
    bc = boxcounting()
    bc.occupation("data/"+data_file, max_level, size = 100)
    fit_show(bc, min_index = min_level)
def plot3d(data_file):
    # Plot
    data = np.loadtxt("data/" + data_file)
    data = data[::10]
    x, y, z = data.T
    fig = go.Figure(data=[go.Scatter3d(x=x, y=y, z=z,
                                   mode='markers', marker=dict(size = 1))])
    fig.write_html("data/lorenz.html")
#    fig.show()
    return 

if __name__ == "__main__":
    data_file = "lorenz.txt"
    tmax = 1000
    n = int(1e7)
    max_level = 10
    min_level = 1
#    write_data(tmax, n, data_file)
#    plot3d(data_file)
    compute_dimension(data_file, max_level, min_level)
    
