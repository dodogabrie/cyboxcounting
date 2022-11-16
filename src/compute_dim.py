import sys
sys.path.append('../')
import numpy as np
import time
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from build.boxcounting import boxcounting

def compute_dim(bc):
    num = np.log2(bc.occ)
    den = np.arange(bc.max_level)# - np.log2(bc.eps)
    print("List of dim:", num/den)
    return num, den

def fit_dim(bc, min_index = 1, max_index = 0):
    if max_index == 0: max_index = bc.max_level
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
    if max_index == 0: max_index = bc.max_level
    f, popt, D, var, y, den = fit_dim(bc, min_index, max_index)
    dim = y/den
    x_array = np.linspace(np.min(den), np.max(den), 100)
    print(f'D = {D} pm {var}')
    print(f'Last D evaluated: {dim[-1]}')

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
    n_rel = bc.n_data/bc.occ[min_index:max_index]
    axs[1].scatter(den, n_rel, c='k')
    axs[1].plot(x_array, np.ones(len(x_array)), linestyle='--', c='r')
    axs[1].set_yscale('log')
    axs[1].set_xlabel(r'$\log(1/\epsilon)$', fontsize = 20)
    axs[1].set_ylabel(r'$\left<n(\epsilon)\right>$', fontsize = 20)
    axs[1].set_title('# medio dati in un quadrato', fontsize = 20)
    plt.show()
    return dim, var

if __name__ == "__main__":
    pass
