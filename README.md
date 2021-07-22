# KMeansDbscanPCA

## Objective
An Unsupervised Learning clustering model, implementing KMeans and Density-Based Spatial Clustering of Application with Noise (DBSCAN) after reducing the dimensionality using Principal Component Analysis. Apply techniques to find similar characteristics of different US counties for predatory marketing, government campaigns, business development, etc by examing 34 attributes.

K-Means and Hierarchical Clustering both fail in creating clusters of arbitrary shapes.


## Model and Vizualization
KMeans:

![](https://github.com/jf20541/KMeansDbscanPCA/blob/main/plots/KMeansPCA2.png?raw=true)

<details>
  <summary>Finding Optimal K using Elbow Method</summary>
  
  ![](https://github.com/jf20541/KMeansDbscanPCA/blob/main/plots/Kmeans_Elbow.png?raw=true)
</details>

DBSCAN: 


<details>
  <summary>Finding Optimal Epsilon using K-Nearest Neighbor</summary>
 
  ![](https://github.com/jf20541/KMeansDbscanPCA/blob/main/plots/optimal_epsilon.png?raw=true)
</details>

PCA: 

![](https://github.com/jf20541/KMeansDbscanPCA/blob/main/plots/pca_explained_var.png?raw=true)
<details>
  <summary>PCA Explained Variance Ratio for N-Components</summary>
 
  ![](https://github.com/jf20541/KMeansDbscanPCA/blob/main/plots/pca_explained_barchart.png?raw=true)
</details>

## Code
- `kmean.py:` Seeked the optimal k-cluster for the population of counties based on the selected 7 PCA attributes 
- `dbscan.py:` Seeked the optimal epsilon and minPoint value with the selected 7 PCA attributes.
- `pca.py:` Dimensionality reduction, plotting, and deploying the PCA model
- `data.py:` Cleaned the data
- `config.py:` Defined file paths as global variable
- `population_seg.ipynb:` Exploring data by visualization and feature engineering

## Parameters
### Principal Component Analysis (PCA)
- `n_components:` Number of prinicpal components. Selecting based on  80% explained variance percentage

### DBSCAN
- `eps:` Used KNN to find the optimal Epsilon value. The maximum distance between two samples for one to be considered as in the neighborhood of the other.
- `min_samples:` The number of samples in a neighborhood for a point to be considered as a core point


### KMeans
- `n_clusters:` Find the optima K-value by plotting and using Elbow Method 
- `max_iter:` Maximum number of iterations of the k-means algorithm for a single run
- `n_init:` Number of time the k-means algorithm will be run with different centroid seeds

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
