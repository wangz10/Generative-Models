from collections import Counter, OrderedDict

import numpy as np
from sklearn import preprocessing, neighbors
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from sklearn.metrics.pairwise import pairwise_distances

import seaborn as sns
# pretty colors
COLORS10 = sns.color_palette("tab10", 10).as_hex()
sns.set_palette(COLORS10)


def plot_embed(X_coords, labels, figsize=(6, 6)):
    '''Scatter plot for the coordinates colored by their labels'''
    X_coords = preprocessing.MinMaxScaler(feature_range=(-1, 1)).fit_transform(X_coords[:, :2])

    fig, ax = plt.subplots(figsize=figsize)
    scatter_proxies = []
    colors = [COLORS10[l] for l in labels]
    ax.scatter(X_coords[:, 0], X_coords[:, 1], s=5, c=colors, edgecolor='none')
    for i in range(10):
        scatter_proxy = Line2D([0],[0], ls="none", 
                               c=COLORS10[i], 
                               marker='o', 
                               markersize=5, 
                               markeredgecolor='none')    
        scatter_proxies.append(scatter_proxy)
    
    # Shrink current axis by 20%
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])

    # Put a legend to the right of the current axis
    ax.legend(scatter_proxies, map(str, range(10)), 
              numpoints=1,
              loc='center left', bbox_to_anchor=(1, 0.5))

    fig.tight_layout()
    return ax

def display_mnist_image(x, figsize=None):
    '''x is an array with shape (1, 784)'''
    fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(x.reshape(28, 28), cmap="gray")
    ax.set_axis_off()
    return ax

def display_mnist_images(xs, figsize=None):
    '''xs is a list of arrays with shape (1,784)'''
    n = len(xs)
    canvas = np.empty((28, 28*n))
    for i, x in enumerate(xs):
        # scale to (0, 1)
        x = preprocessing.minmax_scale(x.T).T
        canvas[:, i*28: (i+1)*28] = x[0].reshape(28, 28)
    fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(canvas, cmap="gray")
    ax.set_axis_off()
    return ax

def sample_latent_space(model, n_dims, n=20, sample_type='uniform', scale=True):
    '''Sample the latent space of n_dims, 
    then generate data (images) using model to organize generated 
    data (images) into a squared canvas for plotting.

    model: need to have `generate` method
    n_dims: the dimension of the latent space
    n: number of images along the canvas
    '''
    canvas = np.empty((28*n, 28*n))
    if sample_type == 'uniform':
        zs_mu = np.random.uniform(-3, 3, n_dims * n**2)
    elif sample_type == 'normal':
        zs_mu = np.random.randn(n_dims * n**2)
    zs_mu = zs_mu.reshape(n**2, n_dims)
    
    xs_gen = model.generate(zs_mu)
    c = 0
    for i in range(n):
        for j in range(n):
            x = xs_gen[c]
            if scale:
                x = preprocessing.minmax_scale(x.T).T
            canvas[(n-i-1)*28:(n-i)*28, j*28:(j+1)*28] = x.reshape(28, 28)
            c += 1
    return canvas
