"""
Created on Thu Feb 27 15:46:50 2020
@author: James Bond
"""

import pandas as pd

base = pd.read_csv('plano-saude2.csv')

x = base.iloc[:, 0:1].values
y = base.iloc[:, 1].values

#Regressão Linear Simples
from sklearn.linear_model import LinearRegression
regressor1 = LinearRegression()
regressor1.fit(x, y)
score1 = regressor1.score(x,y)

regressor1.predict([[40]])

import matplotlib.pyplot as plt
plt.scatter(x,y)
plt.plot(x, regressor1.predict(x), color = 'red')
plt.title('Regressão linear')
plt.xlabel('Idade')
plt.ylabel('custo')

#Regressão Polinomial
from sklearn.preprocessing import PolynomialFeatures
poly = PolynomialFeatures(degree = 4)
x_poly = poly.fit_transform(x)

regressor2 = LinearRegression()
regressor2.fit(x_poly, y)
score2 = regressor2.score(x_poly, y)

regressor2.predict(poly.transform(np.array(40).reshape(1, -1)))

plt.scatter(x,y)
plt.plot(x, regressor2.predict(poly.fit_transform(x)), color = 'red')
plt.title('Regressão Polinomial')
plt.xlabel('Idade')
plt.ylabel('custo')
