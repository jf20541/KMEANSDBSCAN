# KMeansDbscanPCA

## Objective
An Unsupervised Learning clustering model, implementing KMeans and Density-Based Spatial Clustering of Application with Noise (DBSCAN) after reducing the dimensionality using Principal Component Analysis. Apply techniques to find similar characteristics of different US counties for predatory marketing, government campaigns, business development, etc by examing 34 attributes.

## Code
- `kmean.py:` Seeked the optimal k-cluster for the population of counties based on the selected 7 PCA attributes 
- `dbscan.py:` Seeked the optimal epsilon and minPoint value with the selected 7 PCA attributes.
- `pca.py:` Dimensionality reduction, plotting, and deploying the PCA model
- `data.py:` Cleaned the data
- `config.py:` Defined file paths as global variable
- `population_seg.ipynb:` Exploring data by visualization and feature engineering

## Model and Vizualization
### KMeans
K-Means finds the optimal centroids (number of clusters is represented by K) by assigning data points to clusters based on the defined centroids using Elbow Method. K-Means is sensitive to outliers and the number of dimensions increases its scalability decreases.
- `n_clusters:` Find the optima K-value by plotting and using Elbow Method 
- `max_iter:` Maximum number of iterations of the k-means algorithm for a single run
- `n_init:` Number of time the k-means algorithm will be run with different centroid seeds

<details>
  <summary>Finding Optimal K (Elbow Method) Plot</summary>
  
  ![](https://github.com/jf20541/KMeansDbscanPCA/blob/main/plots/Kmeans_Elbow.png?raw=true)
</details>

![](https://github.com/jf20541/KMeansDbscanPCA/blob/main/plots/KMeansPCA2.png?raw=true)


### DBSCAN
An unsupervised algorithm for density-based clustering that identifies distinctive clusters within a high point density which can signal outliers natively. The model has two hyper-parameters Epsilon and Minimum Points. Epsilon is the radius of the neighborhood around any point. Minimum Point is the minimum number of points within the Epsilon radius.

- `eps:` Used KNN to find the optimal Epsilon value. The maximum distance between two samples for one to be considered as in the neighborhood of the other
- `min_samples:` The number of samples in a neighborhood for a point to be considered as a core point


<details>
  <summary>Finding Optimal Epsilon using K-Nearest Neighbor Plot</summary>
 
  ![](https://github.com/jf20541/KMeansDbscanPCA/blob/main/plots/optimal_epsilon.png?raw=true)
</details>

### Principal Component Analysis (PCA)
A method for reducing the dimensionality of a dataset [3220, 34]. With 34 features, it can cause more processing time and noise. Using the explained variance ratio (percentage of variance explained by each of the selected components) to select the number of principal components.

- `n_components:` Number of prinicpal components. Selecting based on  80% explained variance percenta
<details>
  <summary>PCA Explained Variance Ratio for N-Components Plot</summary>
 
  ![](https://github.com/jf20541/KMeansDbscanPCA/blob/main/plots/pca_explained_barchart.png?raw=true)
</details>

![](https://github.com/jf20541/KMeansDbscanPCA/blob/main/plots/pca_explained_var.png?raw=true)

## Data
[Population Segmentation Data](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/XTXCYD) and 
[US Census Data](https://data.census.gov/cedsci/) 
```
'TotalPop', 'Men', 'Women', 'Hispanic', 'White', 'Black', 'Native',
'Asian', 'Pacific', 'Citizen', 'Income', 'IncomeErr', 'IncomePerCap',
'IncomePerCapErr', 'Poverty', 'ChildPoverty', 'Professional', 'Service',
'Office', 'Construction', 'Production', 'Drive', 'Carpool', 'Transit',
'Walk', 'OtherTransp', 'WorkAtHome', 'MeanCommute', 'Employed',
'PrivateWork', 'PublicWork', 'SelfEmployed', 'FamilyWork', 'Unemployment'
```
