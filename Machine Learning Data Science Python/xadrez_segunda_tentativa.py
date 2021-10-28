# -*- coding: utf-8 -*-
"""
Created on Thu Feb  6 11:01:54 2020

@author: James Bond
"""

import pandas as pd

base = pd.read_csv('games.csv')

base.drop('id', 1, inplace=True)
base.drop('created_at', 1, inplace=True)
base.drop('last_move_at', 1, inplace=True)
base.drop('white_id', 1, inplace=True)
base.drop('black_id', 1, inplace=True)
base.drop('moves', 1, inplace=True)


previsores = base.iloc[:,[0,1,2,4,5,6,7,8,9]].values
classe = base.iloc[:, 3].values

classe
previsores
#rated, turns, victory_status, winner, incremente_code, white_rating, black_rating, opening_eco, opening_name, opening_ply
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_previsores = LabelEncoder()
previsores[:, 0] = labelencoder_previsores.fit_transform(previsores[:, 0])
previsores[:, 2] = labelencoder_previsores.fit_transform(previsores[:, 2])
previsores[:, 3] = labelencoder_previsores.fit_transform(previsores[:, 3])
previsores[:, 4] = labelencoder_previsores.fit_transform(previsores[:, 4])
previsores[:, 7] = labelencoder_previsores.fit_transform(previsores[:, 7])
previsores[:, 8] = labelencoder_previsores.fit_transform(previsores[:, 8])


onehotencoder = OneHotEncoder(categorical_features = [0,2,3,4,7,8])
previsores = onehotencoder.fit_transform(previsores).toarray()

labelencoder_classe = LabelEncoder()
classe = labelencoder_classe.fit_transform(classe)

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
previsores = scaler.fit_transform(previsores)

from sklearn.model_selection import train_test_split
previsores_treinamento, previsores_teste, classe_treinamento, classe_teste = train_test_split(previsores, classe, test_size=0.25, random_state=0)

from sklearn.neural_network import MLPClassifier
classificador = MLPClassifier(verbose = True,
                              max_iter=1000,
                              tol = 0.0000010,
                              solver = 'adam',
                              hidden_layer_sizes=(100),
                              activation='relu')
classificador.fit(previsores_treinamento, classe_treinamento)
previsores = classificador.predict(previsores_teste)

from sklearn.metrics import confusion_matrix, accuracy_score
precisão = accuracy_score(classe_teste, previsores)
matriz = confusion_matrix(classe_teste, previsores)