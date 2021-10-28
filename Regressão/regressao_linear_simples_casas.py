"""
Created on Thu Feb 27 10:09:52 2020
@author: James Bond
"""

import pandas as pd

base = pd.read_csv('Penis.csv')

X = base.iloc[:, 1:2].values
Y = base.iloc[:, 2].values

from sklearn.model_selection import train_test_split
X_treinamento, X_teste, Y_treinamento, Y_teste = train_test_split(X,Y,
                                                                  test_size = 0.3,
                                                                  random_state = 0)
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_treinamento, Y_treinamento)
score = regressor.score(X_treinamento, Y_treinamento)

import matplotlib.pyplot as plt
plt.scatter(X_treinamento, Y_treinamento)
plt.plot(X_treinamento, regressor.predict(X_treinamento), color = 'red')

previsoes = regressor.predict(X_teste)

resultado = abs(Y_teste - previsoes)
resultado.mean()

from sklearn.metrics import mean_absolute_error, mean_squared_error
mae = mean_absolute_error(Y_teste, previsoes)
mse = mean_squared_error(Y_teste, previsoes)

plt.scatter(X_teste, Y_teste)
plt.plot(X_teste, regressor.predict(X_teste), color = 'red')

regressor.score(X_teste, Y_teste)