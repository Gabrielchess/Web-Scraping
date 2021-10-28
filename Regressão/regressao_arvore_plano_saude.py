# -*- coding: utf-8 -*-
"""
Created on Sun Mar  1 20:19:13 2020

@author: James Bond
"""

import pandas as pd 

base = pd.read_csv('plano-saude2.csv')

x = base.iloc[:, 0:1].values
y = base.iloc[:, 1].values

from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor()
regressor.fit(x,y)
score = regressor.score(x,y)

import matplotlib.pyplot as plt
plt.scatter(x,y)
plt.plot(x, regressor.predict(x), color = 'red')
plt.title('Regressão com Arvores')
plt.xlabel('Idade')
plt.ylabel('custo')

import numpy as np
x_teste = np.arange(min(x), max(x), 0.1)
x_teste = x_teste.reshape(-1, 1)
plt.scatter(x,y)
plt.plot(x_teste, regressor.predict(x_teste), color = 'red')
plt.title('Regressão com Arvores')
plt.xlabel('Idade')
plt.ylabel('custo')

regressor.predict([[40]])