# -*- coding: utf-8 -*-
"""
Created on Mon Nov 20 14:36:39 2017

@author: Jones
"""
import pandas as pd

base = pd.read_csv('risco_credito.csv')
previsores = base.iloc[:,0:3].values
classe = base.iloc[:,3].values
                  
from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()
previsores[:,0] = labelencoder.fit_transform(previsores[:,0])
previsores[:,1] = labelencoder.fit_transform(previsores[:,1])
previsores[:,2] = labelencoder.fit_transform(previsores[:,2])
                 
from sklearn.naive_bayes import GaussianNB
classificador = GaussianNB()
classificador.fit(previsores, classe)
# história boa, dívida alta, garantias nenhuma, renda > 35
# história ruim, dívida alta, garantias adequada, renda < 15
resultado = classificador.predict([[0,0,1,2], [3, 0, 0, 0]])
print(classificador.classes_)
print(classificador.class_count_)
print(classificador.class_prior_)