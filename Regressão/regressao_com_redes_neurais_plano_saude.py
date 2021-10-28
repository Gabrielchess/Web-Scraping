# -*- coding: utf-8 -*-
"""
Created on Fri Mar  6 11:36:12 2020

@author: James Bond
"""

import pandas as pd

base = pd.read_csv('Penis.csv')

X = base.iloc[:, 11:12].values
y = base.iloc[:, 13].values

from sklearn.preprocessing import StandardScaler
scaler_x = StandardScaler()
X = scaler_x.fit_transform(X)
scaler_y = StandardScaler()
y = scaler_y.fit_transform(y)

from sklearn.neural_network import MLPRegressor
regressor = MLPRegressor()
regressor.fit(X,y)

regressor.score(X,y)

import matplotlib.pyplot as plt
plt.scatter(X,y)
plt.plot(X, regressor.predict(X), color = 'red')
plt.title('Regressão com Redes Neurais')
plt.xlabel('Idade')
plt.ylabel('custo')

previsao = scaler_y.inverse_transform(regressor.predict(scaler_x.transform([[40]])))