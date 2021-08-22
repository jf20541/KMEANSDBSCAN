import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
import config


class PCAModel:
    def __init__(self, data):
        self.data = data

    def plot_pca_var(self):
        # initiated PCA to find optimal components that explain 80% of variance
        pca = PCA(n_components=35)
        pca.fit(self.data)
        variance_exp_cumsum = pca.explained_variance_ratio_.cumsum().round(2)
        fig, axes = plt.subplots(1, 1, figsize=(16, 7), dpi=100)
        xi = np.arange(1, 36, step=1)
        plt.plot(xi, variance_exp_cumsum, marker="o", linestyle="-", color="b")
        plt.title("N-Components for Explained Variance")
        plt.xlabel("Number of Components")
        plt.ylabel("Cumulative variance (%)")
        plt.axhline(y=0.80, color="r", linestyle="-")
        plt.savefig("../plots/pca_explained_var.png")
        plt.show()

    def plot_pca_bar(self):
        # plot bar-chart explaing each components explained variance ratio
        pca = PCA(n_components=15)
        pca.fit(self.data)
        ratio = pca.explained_variance_ratio_
        fig, axes = plt.subplots(1, 1, figsize=(16, 7), dpi=100)
        xi = np.arange(1, 15, step=1)
        features = range(15)
        plt.bar(features, ratio, color="black")
        plt.title("Explained Variance")
        plt.xlabel("Number of Components")
        plt.ylabel("Variance (%)")
        plt.xticks(features)
        plt.savefig("../plots/pca_explained_barchart.png")
        plt.show()


if __name__ == "__main__":
    df = pd.read_csv(config.TESTING_FILE)
    df = df.set_index("State_City")
    plot_model = PCAModel(df)
    plot_model.plot_pca_var()
    plot_model.plot_pca_bar()
