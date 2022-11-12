import numpy as np
import time
import matplotlib.pyplot as plt
from build.boxcounting import boxcounting
from scipy.optimize import curve_fit
np_or = np.logical_or
np_and = np.logical_and

def test_boxcounting():
    n_data = int(3e3)
    max_level = 4
    #data = example_f(n_data)
    TT = []
    data = square(10, N = n_data)
    for level in range(1, max_level):
        bc = boxcounting(data.shape[1])
        start = time.time()
        occ = bc.occupation(data, level)
        t_lev = time.time()-start
        TT.append(t_lev)
        print(f'Tree max deep {level}: {t_lev} sec')
        bc.free()

    plt.scatter([i + 1 for i in range(len(TT))], TT)
    plt.show()
    bc = boxcounting(data.shape[1])
    occ = bc.occupation(data, max_level)
    bc.fit_show(min_index = 1)
    return

def example_f(n):
    x = np.linspace(0, np.pi * 2, n)
    return np.column_stack((x, x**2))

def square(l, N = 50):
    x = np.linspace(0, l, N).astype(float)
    y = np.linspace(0, l, N).astype(float)
    X, Y = np.meshgrid(x, y)
    X = X.flatten()
    Y = Y.flatten()
#    mask_x = np_or(X==np.min(X), X==np.max(X))
#    mask_y = np_or(Y==np.min(Y), Y==np.max(Y))
#    x_sq = X[np.logical_or(mask_x, mask_y)]
#    y_sq = Y[np.logical_or(mask_x, mask_y)]
    data = np.column_stack((X, Y))
#    plt.scatter(X[::10], Y[::10])
#    plt.show()
#    print(len(data))
    return data

if __name__ == '__main__':
    test_boxcounting()
