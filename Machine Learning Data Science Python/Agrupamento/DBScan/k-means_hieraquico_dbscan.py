# -*- coding: utf-8 -*-
"""
Created on Mon Mar 16 13:34:57 2020

@author: Hacker James Bond
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn import datasets

x,y = datasets.make_moons(n_samples = 1500, noise = 0.09)
plt.scatter(x[:, 0], x[:, 1], s = 5)

cores = np.array(['red','blue'],)

kmeans = KMeans(n_clusters = 2)
previsoes = kmeans.fit_predict(x)
plt.scatter(x[:,0], x[:,1], s=5, color = cores[previsoes])

hc = AgglomerativeClustering(n_clusters = 2, affinity = 'euclidean', linkage = 'ward')
previsoes = hc.fit_predict(x)
plt.scatter(x[:,0],x[:,1],s=5, color = cores[previsoes])

dbscan = DBSCAN(eps = 0.1, min_samples = 2)
previsoes = dbscan.fit_predict(x)
plt.scatter(x[:,0], x[:,1], s=5, color = cores[previsoes])
