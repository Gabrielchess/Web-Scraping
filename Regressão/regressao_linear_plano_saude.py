"""
Created on Thu Feb 27 09:12:01 2020
@author: James Bond
"""

import pandas as pd
import numpy as np

base = pd.read_csv('plano_saude.csv')

X = base.iloc[:, 0].values
Y = base.iloc[:, 1].values

correlacao = np.corrcoef(X, Y)
X = X.reshape(-1,1)

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X, Y)

#b0
regressor.coef_

#b1
regressor.intercept_

import matplotlib.pyplot as plt
plt.scatter(X, Y)
plt.title('Regressao Linear simples')
plt.plot(X, regressor.predict(X))
plt.xlabel('Idade')
plt.ylabel('Custo')

previsao1 = regressor.predict([[40]])
previsao2 = regressor.intercept_ + regressor.coef_ * 40

score = regressor.score(X, Y)
    
from yellowbrick.regressor import ResidualsPlot
visualizador = ResidualsPlot(regressor)
visualizador.fit(X, Y)
visualizador.poof()






