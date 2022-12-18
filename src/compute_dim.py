import sys
import os
sys.path.append('../')
import numpy as np
import time
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from build.boxcounting import boxcounting


def create_tree(fname, max_level, min_level = 1, num_tree = 1, size = None):
    if os.path.isfile(fname):
        multi = False
    elif os.path.isdir(fname):
        multi = True 
    else:
        raise Exception(f"No such file or directory: {fname}")
    bc = boxcounting()
    if multi:
        folderfiles = os.listdir(fname)
        bc.set_data_file(fname + folderfiles[0])
        bc.initialize(max_level, size = size, num_tree = num_tree)
        bc.fill_tree()
        for i, file in enumerate(folderfiles[1:], 1):
            print(i, end = '\r')
            bc.set_data_file(fname + file)
            bc.fill_tree()
    else:
        bc.set_data_file(fname)
        bc.initialize(max_level, size = size, num_tree = num_tree)
        bc.fill_tree()
    bc.count_occupation()
    return bc

def save_output(bc, filename, data_dir):
    N = bc.occ
    eps = bc.eps
    np.savetxt(filename, np.column_stack((N, eps)))
    return

def compute_dim(bc = None, file = None):
    if bc != None:
        occ = bc.occ 
        eps = bc.eps
    elif file != None:
        occ, eps = np.loadtxt(file, unpack=True)
    elif bc != None and file != None:
        raise Exception("Please pass a tree or a data file (not both!)")
    else: raise Exception("Please pass a tree or a data file with occ and eps")
    num = np.log10(occ)
    den = np.log10(1/eps)#np.arange(bc.max_level) #- np.log2(bc.eps0)
    sorter = np.argsort(den) 
    den = den[sorter]
    num = num[sorter]
    den[den == 0] = 1
    return num, den

def dim_df(file, block_dim = 4, min_index = 1, max_index = -1):
    """ Compute dimension at group of block_dim points"""
    def f(x, m, q):
        return m * x + q
    init = [1., 0]
    num, den = compute_dim(file = file)
    if max_index == -1: max_index = len(num) + 1
    num = num[min_index:max_index]
    den = den[min_index:max_index]
    num_data = len(num)
    list_dim = []
    list_eps = []
    list_Ddim = []
    for i in range(num_data - block_dim):
        x = den[i:i+block_dim]
        y = num[i:i+block_dim]
        list_eps.append(np.mean(x))
        popt, pcov = curve_fit(f, x, y, p0=init)
        list_dim.append(popt[0])
        list_Ddim.append(np.sqrt(np.diag(pcov))[0])
    fig, ax = plt.subplots(1,1, figsize = (7,7))
    ax.scatter(list_eps, list_dim, s = 25)
    ax.errorbar(list_eps, list_dim, yerr=list_Ddim, fmt = '.')
    ax.set_xlabel(r'$\log_{10}(1/\epsilon)$', fontsize = 20)
    ax.set_ylabel(r'$D$', fontsize = 20)
    ax.set_title('$D$ con fit a gruppi di 5 punti vicini', fontsize = 20)
    plt.setp(ax.get_xticklabels(), fontsize=13)
    plt.setp(ax.get_yticklabels(), fontsize=13)
    ax.grid()
    plt.savefig("figures/lorenz_dim/lorenz_group5.png", dpi = 200)
    plt.show()

def fit_dim(bc, min_index = 1, max_index = 0):
    if max_index == 0: max_index = bc.max_level * bc.num_tree
    if isinstance(bc, str):
        y, den = np.loadtxt(bc, unpack=True)
    else:    
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

def fit_show(bc, min_index = 1, max_index = 0, single_dim = False):
    if max_index == 0: max_index = bc.max_level*bc.num_tree
    min_index = min_index * bc.num_tree
    f, popt, D, var, y, den = fit_dim(bc, min_index, max_index)
    dim = y/den
    all_num, all_den = compute_dim(bc)
    x_array = np.linspace(np.min(den), np.max(den), 100)
    print(f'D = {D} pm {var}')
    # Plot dei risultati
    if single_dim:
        fig, axs = plt.subplots(1,2,figsize=(14,6))
        for ax in axs: 
            ax.grid()
            plt.setp(ax.get_xticklabels(), fontsize=13)
            plt.setp(ax.get_yticklabels(), fontsize=13)
#        axs[0].plot(x_array, f(x_array, *popt))
        axs[0].scatter(den, y, c='k')
        axs[0].set_xlabel(r'$\log_{10}(1/\epsilon)$', fontsize = 20)
        axs[0].set_ylabel(r'$\log_{10}(N(\epsilon))$', fontsize = 20)
#        axs[0].set_title(fr'$y = mx + q$ $\rightarrow$ m = {D:.3f} $\pm$ {var:.3f}', fontsize = 20)
        axs[0].set_title(r'$\log_{10}(N(\epsilon))$ vs $\log_{10}(1/\epsilon)$', fontsize = 20)
        yy = y/den 
        axs[1].scatter(den, yy)
        axs[1].set_xlabel(r'$\log_{10}(1 /\epsilon)$', fontsize = 20)
        axs[1].set_ylabel('D', fontsize = 20)
        axs[1].set_title(r'D al variare di $\log_{10}(1 /\epsilon)$', fontsize = 20)
    else:
        fig, axs = plt.subplots(1,1,figsize=(7,7))
        axs.plot(x_array, f(x_array, *popt))
        axs.grid()
        plt.setp(axs.get_xticklabels(), fontsize=13)
        plt.setp(axs.get_yticklabels(), fontsize=13)
        axs.scatter(den, y, c='k')
        axs.set_xlabel(r'$\log_{10}(1/\epsilon)$', fontsize = 20)
        axs.set_ylabel(r'$\log_{10}(N(\epsilon))$', fontsize = 20)
#        axs.set_title(r'$\log_{10}(N(\epsilon))$ vs $\log_{10}(1/\epsilon)$', fontsize = 20)
        axs.set_title(fr'$y = mx + q$ $\rightarrow$ m = {D:.3f} $\pm$ {var:.3f}', fontsize = 20)
    return dim, var, fig, axs

if __name__ == "__main__":
    pass
