import random
import matplotlib.pyplot as plt
import math
import numpy as np
import anndata
import pandas as pd
import scvelo as scv
import cellrank as cr


def generate_data(n_points, n_classes, func, x_range, sigma, overlap_coeff, n_dims=2, show=False):
    assert x_range[0] < x_range[1]
    data = np.zeros((n_points, n_dims)) # +1 for class assignment
    classes = np.zeros((n_points))

    def _assign_class(index, n_points, n_classes):
        pts_per_class = np.round(n_points / n_classes)
        if index // pts_per_class < n_classes:
            return index // pts_per_class
        else:
            return n_classes - 1


    for i, x in enumerate(np.linspace(x_range[0], x_range[1], n_points)):
        shift = random.gauss(0, 0.31) * overlap_coeff + x
        x += shift

        rand_angle = random.randint(0, 359) / 360
        rand_dist = random.gauss(0, sigma)
        new_x = (x) + rand_dist*math.cos(rand_angle)
        new_y = func(x) + rand_dist*math.sin(rand_angle)
        data[i, 0] = new_x
        data[i, 1] = new_y
        classes[i] = _assign_class(i, n_points, n_classes)

    if show and n_dims == 2:
        plt.scatter(data[:, 0], data[:, 1], c=classes, s=1, alpha=0.62)
        plt.show()

    # Format as anndata object
    # Note: this will not work for larger data sets.
    obs = pd.DataFrame({"time_point" : classes}, index=["Cell %d " % i for i in range(n_points)])
    var = pd.DataFrame({"Gene %d" % i for i in range(n_dims)})
    X = data
    adata = anndata.AnnData(X, obs, var)

    return adata



def calculate_pseudotime(adata, cluster_key, initial_vals, final_vals):
    scv.pp.neighbors(adata)
    scv.tl.umap(adata)
    scv.pl.scatter(adata, basis='umap', color='time_point')
    scv.pp.moments(adata)




# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    n_points = 1001
    n_classes = 6
    func = lambda x: (0.3 * x**4) + (0.3 * x**3) - (1.4 * x**2) - (0.9 * x) - 0.1
    x_range = [-1.5, 1]
    sigma = 0.5
    shift_overlap = 0.1

    adata = generate_data(n_points, n_classes, func, x_range, sigma, shift_overlap, show=True)
    calculate_pseudotime(adata, '', '', '')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
