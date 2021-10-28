# -*- coding: utf-8 -*-
"""
Created on Thu Jan 23 11:31:18 2020

@author: James Bond
"""

import pandas as pd

base = pd.read_csv('titanic.csv')

previsores = base.iloc[:, 0:4].values
classe = base.iloc[:, 4].values

from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = 'nan', strategy = 'mean', axis = 0)
imputer = imputer.fit(previsores[:,2])
previsores[:, 2] = imputer.transform(previsores[:, 2])

#categorização das variaves previsoras
#Pclass numerica descritiva - Age numerica descritiva - Sex categorica Nominal
#Vamos transformar essas desgraças em variaveis numéricas caralho
from sklearn.preprocessing import LabelEncoder
labelencoder_previsores = LabelEncoder()
previsores[:, 0] = labelencoder_previsores.fit_transform(previsores[:, 0])
previsores[:, 3] = labelencoder_previsores.fit_transform(previsores[:, 3])

onehotencoder = OneHotEncoder(categorical_features = [0,1,3])
previsores = onehotencoder.fit_transform(previsores).toarray()

base.describe()
base.loc[pd.isnull(base['Age'])]