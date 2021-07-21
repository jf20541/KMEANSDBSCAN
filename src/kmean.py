import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import config
import numpy as np
import matplotlib.pyplot as plt

# class KMeansPlot:
#     def __init__(self, data):
#         self.data = data

#     def plot_optimum_cluster(self):
#         #set a list to append the iter values
#         iter_num = []
#         for i in range(1, 15):
#             #perform kmeans to get best cluster value using elbow method
#             model = KMeans(n_clusters = i, max_iter = 500)
#             model.fit(self.data)
#             iter_num.append(model.inertia_)
#             #plot the optimum graph
#         plt.plot(range(1, 15), iter_num)
#         plt.title('Optimal K using Elbow Method')
#         plt.xlabel('Number of K')
#         plt.ylabel('Number of Iterations')
#         plt.savefig('../plots/Kmeans_Elbow.png')
#         plt.show()


class KMeansPCA:
    def __init__(self, data):
        self.data = data

    def plot_pca_kmeans(self):
        pca = PCA(n_components=15).fit(self.data)
        pca_transform = pca.transform(self.data)
        kmeans = KMeans(n_clusters=15)
        y_kmeans = kmeans.fit(pca_transform)
        labels = y_kmeans.labels_
        for i in range(0, 6):
            color = ["red", "blue", "green", "cyan", "yellow", "black"]
            label = [
                "cluster 1",
                "cluster 2",
                "cluster 3",
                "cluster 4",
                "cluster 5",
                "cluster 6",
            ]
            plt.scatter(
                np.where(labels == i),
                self.data.iloc[labels == i, 0],
                s=25,
                c=color[i],
                label=label[i],
            )
        # plt.title('frequency clustering w.r.t damping and eigen modes[PCA_KMEANS]')
        plt.xlabel("sample index")
        plt.ylabel("frequencies")
        plt.legend()
        plt.show()


if __name__ == "__main__":
    df = pd.read_csv(config.TESTING_FILE)
    df = df.set_index("State_City")

    # plot = KMeansPlot(df)
    # plot.plot_optimum_cluster()

    model = KMeansPCA(df)
    model.plot_pca_kmeans()
