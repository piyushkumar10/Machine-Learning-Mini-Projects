import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('Ads_CTR_Optimisation.csv')

import random

n = 10000
d = 10
selected = []
reward = 0
for i in range(0, n):
    ad = random.randrange(10)
    selected.append(ad)
    tempreward = dataset.values[i, ad]
    reward = reward + tempreward
plt.hist(selected)
plt.xlabel('Ads')
plt.ylabel('No. of times selected')
plt.show()
