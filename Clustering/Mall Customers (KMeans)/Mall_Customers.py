import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('Mall_Customers.csv')  # import data
x = dataset.iloc[:, [3, 4]].values

# Using WCSS
from sklearn.cluster import KMeans

wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
    kmeans.fit(x)
    wcss.append(kmeans.inertia_)

plt.plot(range(1, 11), wcss)
plt.title("WCSS Graph")
plt.xlabel("Clusters")
plt.ylabel('WCSS')
plt.show()

# KMeans
kmeans = KMeans(n_clusters=5, init='k-means++', max_iter=300, n_init=10, random_state=0)
clustered = kmeans.fit_predict(x)

plt.scatter(x[clustered == 0, 0], x[clustered == 0, 1], s=50, c='red', label="Cluster 1")
plt.scatter(x[clustered == 1, 0], x[clustered == 1, 1], s=50, c='green', label="Cluster 2")
plt.scatter(x[clustered == 2, 0], x[clustered == 2, 1], s=50, c='cyan', label="Cluster 3")
plt.scatter(x[clustered == 3, 0], x[clustered == 3, 1], s=50, c='magenta', label="Cluster 4")
plt.scatter(x[clustered == 4, 0], x[clustered == 4, 1], s=50, c='blue', label="Cluster 5")
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=300, c='yellow', label='Centroids')

plt.title("Clusters")
plt.xlabel('Annual Income')
plt.ylabel("Spending")
plt.legend()
plt.show()
