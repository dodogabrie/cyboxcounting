import sys
sys.path.append('../')
import numpy as np
import time
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from build.boxcounting import boxcounting


def save_output(bc, filename, data_dir):
    N = bc.occ
    eps = bc.eps
    np.savetxt(filename, np.column_stack((N, eps)))
    return

def compute_dim(bc = None, file = None):
    if file == None:
        num = np.log2(bc.occ)
        den = np.log2(1/bc.eps)#np.arange(bc.max_level) #- np.log2(bc.eps0)
        sorter = np.argsort(den) 
        den = den[sorter]
        num = num[sorter]
        for i in range(bc.num_tree):
            den[i] = 1
    else:
        occ, eps = np.loadtxt(file, unpack=True)
        num = np.log2(occ)
        den = np.log2(1/eps)#np.arange(bc.max_level) #- np.log2(bc.eps0)
        sorter = np.argsort(den) 
        den = den[sorter]
        num = num[sorter]
        den[den == 0] = 1

#    print("Original Radius:", bc.eps0)
#    print("List of dim:", num/den)
    return num, den

def dim_df(bc = None, file = None, block_dim = 4):
    """ Compute dimension at group of block_dim points"""
    def f(x, m, q):
        return m * x + q
    init = [1., 0]
    num, den = compute_dim(bc = bc, file = file)
    print(num, den)
    num_data = len(num)
    list_dim = []
    list_eps = []
    for i in range(num_data - block_dim):
        x = den[i:i+block_dim]
        y = num[i:i+block_dim]
        list_eps.append(np.mean(x))
        popt, pcov = curve_fit(f, x, y, p0=init)
        list_dim.append(popt[0])
    plt.plot(list_eps, list_dim)
    plt.show()

def fit_dim(bc, min_index = 1, max_index = 0):
    if max_index == 0: max_index = bc.max_level * bc.num_tree
    y, den = compute_dim(bc)
    y = y[min_index:max_index]
    den = den[min_index:max_index]
    def f(x, m, q):
        return m * x + q
    init = [1., 0]
    popt, pcov = curve_fit(f, den, y, p0=init)
    x_array = np.linspace(np.min(den), np.max(den), 100)
    D = popt[0]
    var = np.sqrt(np.diag(pcov))[0]
    return f, popt, D, var, y, den

def fit_show(bc, min_index = 1, max_index = 0):
    if max_index == 0: max_index = bc.max_level*bc.num_tree
    min_index = min_index * bc.num_tree
    f, popt, D, var, y, den = fit_dim(bc, min_index, max_index)
    dim = y/den
    x_array = np.linspace(np.min(den), np.max(den), 100)
    print(f'D = {D} pm {var}')
    print(den)
    print(y)
    # Plot dei risultati
    fig, axs = plt.subplots(1,2,figsize=(15,7))
    plt.suptitle(r'Fattore di scala: $\epsilon = 2^n$', fontsize = 20)
    for ax in axs: 
        ax.grid()
        plt.setp(ax.get_xticklabels(), fontsize=13)
        plt.setp(ax.get_yticklabels(), fontsize=13)
    axs[0].plot(x_array, f(x_array, *popt))
    axs[0].scatter(den, y, c='k')
    axs[0].set_xlim(np.min(den)-1,np.max(den)+1)
    axs[0].set_ylim(np.min(y)-1,np.max(y)+1)
    axs[0].set_xlabel(r'$\log(1/\epsilon)$', fontsize = 20)
    axs[0].set_ylabel(r'$\log(N(\epsilon))$', fontsize = 20)
    axs[0].set_title(fr'$y = mx$ $\rightarrow$ m = {D:.3f} $\pm$ {var:.3f}', fontsize = 20)
    axs[1].scatter(den, dim, c='k')
#    axs[1].plot(x_array, np.ones(len(x_array)), linestyle='--', c='r')
#    axs[1].set_yscale('log')
    axs[1].set_xlabel(r'$\log(1/\epsilon)$', fontsize = 20)
    axs[1].set_ylabel('bc-Dim', fontsize = 20)
    axs[1].set_title('dimensione', fontsize = 20)
    plt.show()
    return dim, var
def dimension_convergence(bc):
    return 

if __name__ == "__main__":
    pass
