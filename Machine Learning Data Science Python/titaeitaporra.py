import pandas as pd
from sklearn import preprocessing

base = pd.read_csv('titanic.csv')
               
previsores = base.iloc[:, 1:4].values
classe = base.iloc[:, 4].values

#ver se tem negativo
base.loc[base['Age']<0]
#apagar a linha de nomes
base.drop('Name', 1, inplace=True)
base.drop(base[base.Age == 'nan'].index, inplace=True)
previsores[:,1]
penis = []

base[base['PClass']] = labelencoder_previsores.fit_transform(base['PClass'])

le = preprocessing.LabelEncoder()
le.fit(previsores[:,0])

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_previsores = LabelEncoder()
previsores[:, 0] = labelencoder_previsores.fit_transform(previsores[:, 0])
previsores[:, 2] = labelencoder_previsores.fit_transform(previsores[:, 2])

onehotencoder = OneHotEncoder(categorical_features = [0,2])
previsores = onehotencoder.fit_transform(previsores).toarray()

oi = previsores[:, 0] 

oi





from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
previsores = scaler.fit_transform(previsores)

from sklearn.model_selection import train_test_split
previsores_treinamento, previsores_teste, classe_treinamento, classe_teste = train_test_split(previsores, classe, test_size=0.25, random_state=0)

# importação da biblioteca
from sklearn.neighbors import KNeighborsClassifier
# criação do classificador
classificador = KNeighborsClassifier(n_neighbors=5, metric = 'minkowski', p = 2)
classificador.fit(previsores_treinamento, classe_treinamento)
previsores = classificador.predict(previsores_teste)

from sklearn.metrics import confusion_matrix, accuracy_score
precisão = accuracy_score(classe_teste, previsores)
matriz = confusion_matrix(classe_teste, previsores)
