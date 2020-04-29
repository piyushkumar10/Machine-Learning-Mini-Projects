import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv("Mall_Customers.csv")  # import datasets
x = dataset.iloc[:, [3, 4]].values

import scipy.cluster.hierarchy as sch

dendrogram = sch.dendrogram(sch.linkage(x, method='ward'))
plt.title('Dendrogram')
plt.xlabel('Customers')
plt.ylabel('Euclidean Distances')
plt.show()

# Clustering

from sklearn.cluster import AgglomerativeClustering

hc = AgglomerativeClustering(n_clusters=5, affinity='euclidean', linkage='ward')
clustered = hc.fit_predict(x)

# Plotting 2d in graphs

plt.scatter(x[clustered == 0, 0], x[clustered == 0, 1], s=50, c='red', label="Cluster 1")
plt.scatter(x[clustered == 1, 0], x[clustered == 1, 1], s=50, c='green', label="Cluster 2")
plt.scatter(x[clustered == 2, 0], x[clustered == 2, 1], s=50, c='cyan', label="Cluster 3")
plt.scatter(x[clustered == 3, 0], x[clustered == 3, 1], s=50, c='magenta', label="Cluster 4")
plt.scatter(x[clustered == 4, 0], x[clustered == 4, 1], s=50, c='blue', label="Cluster 5")

plt.title("Clusters")
plt.xlabel('Annual Income')
plt.ylabel("Spending")
plt.legend()
plt.show()
