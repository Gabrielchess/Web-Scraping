# -*- coding: utf-8 -*-
"""
Created on Thu Mar 19 23:14:13 2020

@author: Hacker James Bond
"""


import pandas as pd

base = pd.read_csv('Penis.csv')

base.drop('track_id', inplace=True, axis=1)
base.drop('album_id', inplace=True, axis=1)
base.drop('release_date', inplace=True, axis=1)
base.drop('Posicao_album', inplace=True, axis=1)
base.drop('type', inplace=True, axis=1)
base.drop('list_of_genres', inplace=True, axis=1)
base.drop('key', inplace=True, axis=1)
base.drop('mode', inplace=True, axis=1)

X = base.iloc[:, 0:14].values
Y = base.iloc[:, 14].values

from sklearn.model_selection import train_test_split
X_treinamento, X_teste, Y_treinamento, Y_teste = train_test_split(X,Y,
                                                                  test_size = 0.3,
                                                                  random_state = 0)

from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor()
regressor.fit(x,y)
score = regressor.score(x,y)

previsoes = regressor.predict(X_teste)
from sklearn.metrics import mean_absolute_error
mae = mean_absolute_error(Y_teste, previsoes)

regressor.score(X_teste, Y_teste)

regressor.intercept_
regressor.coef_
len(regressor.coef_)