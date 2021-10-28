# -*- coding: utf-8 -*-
"""
Created on Sun Feb 16 17:33:16 2020

@author: James Bond
"""

import pandas as pd

base = pd.read_csv('credit_data.csv')
base.loc[base.age < 0, 'age'] = 40.92
               
previsores = base.iloc[:, 1:4].values
classe = base.iloc[:, 4].values

from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
imputer = imputer.fit(previsores[:, 1:4])
previsores[:, 1:4] = imputer.transform(previsores[:, 1:4])

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
previsores = scaler.fit_transform(previsores)

from sklearn.naive_bayes import GaussianNB

import numpy as np
a = np.zeros(5)
previsores.shape
previsores.shape[0]
b = np.zeros(shape=(previsores.shape[0], 1))

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
kfold = StratifiedKFold(n_splits = 10, shuffle = True, random_state = 0)
resultados = []
for indice_treinamento, indice_teste in kfold.split(previsores,
                                                    np.zeros(shapes=(previsores.shape[0], 1))):
    #print("Indice treinamento: ", indice_treinamento, "Indice teste: ", indice_teste)
    classificador = GaussianNB()
    classificador.fit(previsores[indice_treinamento], classe[indice_treinamento])
    previsoes = classificador.predict(previsores[indice_teste])
    precisao = accuracy_score(classe[indice_teste],previsoes)
    resultados.append(precisao)
    
resultados = np.asanyarray(resultados)
resultados.mean()
resultados.std()