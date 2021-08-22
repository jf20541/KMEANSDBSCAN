import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import config
import numpy as np
import matplotlib.pyplot as plt


class PlotKMeans:
    def __init__(self, data):
        self.data = data

    def pca_model(self):
        # Fit and transformed PCA to 7 components to df
        return PCA(n_components=7).fit_transform(self.data)

    def plot_optimum_cluster(self):
        pca = self.pca_model()
        # set empty list for each K-iterations
        # to find the optimal k-value using Elbow Method
        iter_num = []
        for i in range(1, 10):
            model = KMeans(n_clusters=i, max_iter=500)
            model.fit(pca)
            iter_num.append(model.inertia_)
        plt.plot(range(1, 10), iter_num)
        plt.title("Optimal K using Elbow Method")
        plt.xlabel("Number of K")
        plt.ylabel("Number of Iterations")
        plt.legend()
        plt.savefig("../plots/Kmeans_Elbow.png")
        plt.show()


class KMeansPCA:
    def __init__(self, data):
        self.data = data

    def plot_pca_kmeans(self):
        # initiated PCA and KMeans with defined parameters
        # PCA=7 Components explain 80% and K=6 using Elbow Method
        pca = PCA(n_components=7).fit_transform(self.data)
        model = KMeans(n_clusters=6)
        model_fit = model.fit(pca)
        labels = model_fit.labels_
        # plot each cluster
        for i in range(0, 6):
            color = ["red", "blue", "green", "cyan", "yellow", "black"]
            label = [
                "Cluster1",
                "Cluster2",
                "Cluster3",
                "Cluster4",
                "Cluster5",
                "Cluster6",
            ]
            plt.scatter(
                np.where(labels == i),
                pca[labels == i, 0],
                s=25,
                c=color[i],
                label=label[i],
            )
        plt.title("Frequency Clustering KMeans & PCA")
        plt.xlabel("Sample")
        plt.ylabel("Frequencies")
        plt.savefig("../plots/KMeansPCA.png")
        plt.legend()
        plt.show()


if __name__ == "__main__":
    df = pd.read_csv(config.TESTING_FILE)
    df = df.set_index("State_City")

    plot = PlotKMeans(df)
    plot.plot_optimum_cluster()

    model = KMeansPCA(df)
    model.plot_pca_kmeans()
