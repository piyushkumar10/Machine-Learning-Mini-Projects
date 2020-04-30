import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

dataset = pd.read_csv('Wine.csv')
x = dataset.iloc[:, 0:13].values
y = dataset.iloc[:, 13].values

xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.2, random_state=0)
# Scaling
scaler = StandardScaler()
xtrain = scaler.fit_transform(xtrain)
xtest = scaler.fit_transform(xtest)

# Dimensionality Reduction
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as lda

lda = lda(n_components=2)
xtrain = lda.fit_transform(xtrain, ytrain)
xtest = lda.transform(xtest)

from sklearn.naive_bayes import GaussianNB

classifier = GaussianNB()
classifier.fit(xtrain, ytrain)
classifiedvalue = classifier.predict(xtest)

from sklearn.metrics import confusion_matrix

confusionmatrix = confusion_matrix(ytest, classifiedvalue)
from matplotlib.colors import ListedColormap

# test Set Visualisation
xset, yset = xtest, ytest
xaxis, yaxis = np.meshgrid(np.arange(start=xset[:, 0].min() - 1, stop=xset[:, 0].max() + 1, step=0.01),
                           np.arange(start=xset[:, 1].min() - 1, stop=xset[:, 1].max() + 1, step=0.01))

plt.contourf(xaxis, yaxis, classifier.predict(np.array([xaxis.ravel(), yaxis.ravel()]).T).reshape(xaxis.shape),
             alpha=0.4, cmap=ListedColormap(('red', 'green', 'blue')))

plt.xlim(xaxis.min(), xaxis.max())
plt.ylim(yaxis.min(), yaxis.max())
for i, j in enumerate(np.unique(yset)):
    plt.scatter(xset[yset == j, 0], xset[yset == j, 1], c=ListedColormap(('red', 'green', 'blue'))(i), label=j)
plt.title("Naive Bayes(Test)")
plt.xlabel("LDA1")
plt.ylabel("LDA2")
plt.legend()
plt.show()
