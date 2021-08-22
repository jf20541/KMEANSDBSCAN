import pandas as pd
import config
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN

# import data
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.neighbors import NearestNeighbors

"""
epsilon and minPoints. Epsilon is the radius of the circle to be created around each
data point to check the density and minPoints is the minimum number of 
data points required inside that circle for that data point to be classified as a Core point.
"""


class DBSACNModel:
    def __init__(self, data):
        self.data = data

    def pca_array(self):
        # Fit and transformed PCA to 7 components to df
        return PCA(n_components=7).fit_transform(self.data)

    def find_epsilon(self):
        # instatiate the pca array function
        pca = self.pca_array()
        # instatiate and fitted KNN
        neighbors = NearestNeighbors(n_neighbors=7)
        neighbors_fit = neighbors.fit(pca)
        distances, _ = neighbors_fit.kneighbors(pca)
        distances = np.sort(distances, axis=0)
        distances = distances[:, 1]
        plt.plot(distances)
        plt.title("Optimal Epsilon using KNN")
        plt.ylabel("Epsilon (Distance)")
        plt.xlabel("Data Points sorted by Distance")
        plt.savefig("../plots/optimal_epsilon.png")
        plt.show()

    def plot_dbscan(self):
        # instatiate the pca array function
        pca = self.pca_array()
        # Fit the pca array to DBSCAN with defined parameter
        dbscan = DBSCAN(eps=0.22, min_samples=16).fit(pca)
        core_samples_mask = np.zeros_like(dbscan.labels_)
        core_samples_mask[dbscan.core_sample_indices_] = True
        # Estimate number of clusters (-1) is for outliers
        labels = dbscan.labels_
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        print(f"Estimated number of clusters: {n_clusters}")
        for i in range(0, n_clusters):
            color = ["green", "blue", "red"]
            label = [
                "Cluster1",
                "Cluster2",
                "Cluster3",
            ]
            plt.scatter(
                np.where(labels == i),
                pca[labels == i, 0],
                s=25,
                c=color[i],
                label=label[i],
            )

        plt.title("Frequency Clustering DBSCAN & PCA")
        plt.xlabel("Sample")
        plt.ylabel("Frequencies")
        plt.savefig("../plots/DBSCAN_PCA.png")
        plt.legend()
        plt.show()


if __name__ == "__main__":
    df = pd.read_csv(config.TESTING_FILE)
    df = df.set_index("State_City")
    model = DBSACNModel(df)
    model.find_epsilon()
    model.plot_dbscan()
