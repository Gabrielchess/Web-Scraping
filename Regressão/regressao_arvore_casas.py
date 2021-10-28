# -*- coding: utf-8 -*-
"""
Created on Mon Mar  2 14:56:52 2020

@author: James Bond
"""


import pandas as pd

base = pd.read_csv('house_prices.csv')

X = base.iloc[:, 3:19].values
Y = base.iloc[:, 2].values

from sklearn.model_selection import train_test_split
X_treinamento, X_teste, Y_treinamento, Y_teste = train_test_split(X,Y,
                                                                  test_size = 0.3,
                                                                  random_state = 0)
from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor()
regressor.fit(X_treinamento, Y_treinamento)
score = regressor.score(X_treinamento, Y_treinamento)

previsoes = regressor.predict(X_teste)

from sklearn.metrics import mean_absolute_error
mae = mean_absolute_error(Y_teste, previsoes)

regressor.score(X_teste, Y_teste)