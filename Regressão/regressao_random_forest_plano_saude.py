# -*- coding: utf-8 -*-
"""
Created on Mon Mar  2 15:34:20 2020

@author: James Bond
"""

import pandas as pd

base = pd.read_csv('plano-saude2.csv')

x = base.iloc[:, 0:1].values
y = base.iloc[:, 1].values

from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators = 10)
regressor.fit(x,y)
score = regressor.score(x, y)

import numpy as np
x_teste = np.arange(min(x), max(x), 0.1)
x_teste = x_teste.reshape(-1,1)

import matplotlib.pyplot as plt
plt.scatter(x,y)
plt.plot(x_teste, regressor.predict(x_teste), color = 'red')
plt.title('Regressão com Random Forest')
plt.xlabel('Idade')
plt.ylabel('custo')

previsao = regressor.predict([[40]])