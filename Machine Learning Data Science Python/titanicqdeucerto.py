"""
Created on Fri Jan 24 14:31:50 2020
@author: James Bond
"""
import pandas as pd
import numpy as np

base = pd.read_csv('titanic.csv')
base.drop('Name', 1, inplace=True)
base.isnull().sum()


base.loc[pd.isnull(base['PClass'])]

from sklearn.impute import SimpleImputer
imp = SimpleImputer(missing_values = np.nan, strategy = 'mean')
nova_idade = base['Age'].values
nova_idade = nova_idade.reshape(-1, 1)
nova_idade = imp.fit_transform(nova_idade)
base['Age'] = nova_idade

imp2 = SimpleImputer(missing_values = np.nan, strategy = 'most_frequent')
trapziodecendente = base['PClass'].values
trapziodecendente = trapziodecendente.reshape(-1, 1)
trapziodecendente = imp2.fit_transform(trapziodecendente)
base['PClass'] = trapziodecendente
base.loc[pd.isnull(base['PClass'])]

previsores = base.iloc[:,0:3].values
classe = base.iloc[:, 3].values

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_previsores = LabelEncoder()
previsores[:, 0] = labelencoder_previsores.fit_transform(previsores[:, 0])
previsores[:, 2] = labelencoder_previsores.fit_transform(previsores[:, 2])

onehotencoder = OneHotEncoder(categorical_features = [0,2])
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

import collections
collections.Counter(classe_teste)
