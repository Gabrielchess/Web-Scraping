# -*- coding: utf-8 -*-
"""
Created on Fri Feb 28 12:26:26 2020

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
from sklearn.preprocessing import PolynomialFeatures
poly =  PolynomialFeatures(degree = 4)
X_treinamento_poly = poly.fit_transform(X_treinamento)
X_teste_poly = poly.transform(X_teste)

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_treinamento_poly, Y_treinamento)
score = regressor.score(X_treinamento_poly, Y_treinamento)

previsores = regressor.predict(X_teste_poly)

from sklearn.metrics import mean_absolute_error
mae = mean_absolute_error(Y_teste, previsores)