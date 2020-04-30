import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

dataset = pd.read_csv('Social_Network_Ads.csv')
x = dataset.iloc[:, [2, 3]].values
y = dataset.iloc[:, 4].values

xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.2, random_state=0)
# Scaling
scaler = StandardScaler()
xtrain = scaler.fit_transform(xtrain)
xtest = scaler.fit_transform(xtest)

# Dimensionality Reduction
from sklearn.decomposition import KernelPCA as kpca

kpca = kpca(n_components=2, kernel='rbf')
xtrain = kpca.fit_transform(xtrain)
xtest = kpca.transform(xtest)

from sklearn.naive_bayes import GaussianNB

classifier = GaussianNB()
classifier.fit(xtrain, ytrain)
classifiedvalue = classifier.predict(xtest)

from sklearn.metrics import confusion_matrix

confusionmatrix = confusion_matrix(ytest, classifiedvalue)
from matplotlib.colors import ListedColormap

# test Set Visualisation
xset, yset = xtrain, ytrain
xaxis, yaxis = np.meshgrid(np.arange(start=xset[:, 0].min() - 1, stop=xset[:, 0].max() + 1, step=0.01),
                           np.arange(start=xset[:, 1].min() - 1, stop=xset[:, 1].max() + 1, step=0.01))

plt.contourf(xaxis, yaxis, classifier.predict(np.array([xaxis.ravel(), yaxis.ravel()]).T).reshape(xaxis.shape),
             alpha=0.4, cmap=ListedColormap(('red', 'green')))

plt.xlim(xaxis.min(), xaxis.max())
plt.ylim(yaxis.min(), yaxis.max())
for i, j in enumerate(np.unique(yset)):
    plt.scatter(xset[yset == j, 0], xset[yset == j, 1], c=ListedColormap(('red', 'green'))(i), label=j)
plt.title("Naive Bayes(Test)")
plt.xlabel("KPCA1")
plt.ylabel("KPCA2")
plt.legend()
plt.show()
