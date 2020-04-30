import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math

dataset = pd.read_csv('Ads_CTR_Optimisation.csv')

n = 10000
d = 10
selectedads = []
selections = [0] * d
totalrewards = [0] * d
total = 0

for rounds in range(0, n):
    maxub = 0
    ad = 0
    for i in range(0, d):
        if (selections[i] > 0):
            averagereward = totalrewards[i] / selections[i]
            deli = math.sqrt(3 / 2 * math.log(n + 1) / selections[i])
            ub = averagereward + deli
        else:
            ub = 1e400
        if (ub > maxub):
            maxub = ub
            ad = i
    selectedads.append(ad)
    selections[ad] = selections[ad] + 1
    reward = dataset.values[rounds, ad]
    totalrewards[ad] = totalrewards[ad] + reward
    total = total + reward

plt.hist(selectedads)
plt.xlabel('Ads')
plt.ylabel('No. of times selected')
plt.show()
