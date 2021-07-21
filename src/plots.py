import matplotlib.pyplot as plt
import numpy as np


def plot_pca_var(explained_var):
    variance_exp_cumsum = explained_var.cumsum().round(2)
    fig, axes = plt.subplots(1, 1, figsize=(16, 7), dpi=100)
    xi = np.arange(1, 36, step=1)
    plt.plot(xi, variance_exp_cumsum, marker="o", linestyle="-", color="b")
    plt.title("N-Components for Explained Variance")
    plt.xlabel("Number of Components")
    plt.ylabel("Cumulative variance (%)")
    plt.axhline(y=0.95, color="r", linestyle="-")
    plt.savefig("../plots/pca_explained_var.png")
    plt.show()


# plots.plot_pca_var(pca.explained_variance_ratio_)


def plot_pca_bar(n_components, ratio):
    # Plot the explained variances
    features = range(n_components)
    plt.bar(features, ratio, color="black")
    plt.title("Explained Variance")
    plt.xlabel("Number of Components")
    plt.ylabel("Variance (%)")
    plt.xticks(features)
    plt.savefig("../plots/pca_explained_barchart.png")
    plt.show()
