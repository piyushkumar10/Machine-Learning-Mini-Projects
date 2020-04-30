import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('Restaurant_Reviews.tsv', delimiter='\t', quoting=3)

import re
import nltk

nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

cleaned = []
for i in range(0, 1000):
    cleanedreview = re.sub('[^a-zA-Z]', ' ', dataset['Review'][i])
    cleanedreview = cleanedreview.lower()
    cleanedreview = cleanedreview.split()
    stem = PorterStemmer()
    cleanedreview = [stem.stem(word) for word in cleanedreview if not word in stopwords.words('english')]
    cleanedreview = ' '.join(cleanedreview)
    cleaned.append(cleanedreview)
from sklearn.feature_extraction.text import CountVectorizer

cv = CountVectorizer(max_features=1500)
x = cv.fit_transform(cleaned).toarray()
y = dataset.iloc[:, 1].values
from sklearn.model_selection import train_test_split

xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.2, random_state=0)

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
xtrain = scaler.fit_transform(xtrain)
xtest = scaler.fit_transform(xtest)

from sklearn.naive_bayes import GaussianNB

classifier = GaussianNB()
classifier.fit(xtrain, ytrain)

classifiedvalue = classifier.predict(xtest)

from sklearn.metrics import confusion_matrix

confusionmatrix = confusion_matrix(ytest, classifiedvalue)
