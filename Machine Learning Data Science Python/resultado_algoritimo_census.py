# -*- coding: utf-8 -*-
"""
Created on Sat Jan 18 23:40:35 2020

@author: James Bond
"""

Census
___________________________________
0.7559 - Base line classifier (ZeroR)

0.4767 - Naive Bayes (labelencoder + onehotencoder + escalonamento)
0.7952 - Naive Bayes (labelencoder)
0.7950 - Naive Bayes (labelencoder + onehotencoder)
0.8057 - Naive Bayes (labelencoder + escalonamento)

0.8102 - Árvore de decisão (labelencoder + onehotencoder + escalonamento)
0.8128 - Árvore de decisão (labelencoder)
0.8102 - Árvore de decisão (labelencoder + onehotencoder)
0.8128 - Árvore de decisão (labelencoder + escalonamento)

0.8476 - Random Forest n_estimators = 40 (labelencoder + onehotencoder + escalonamento)
0.8481 - Random Forest n_estimators = 40 (labelencoder)
0.8489 - Random Forest n_estimators = 40 (labelencoder + onehotencoder)
0.8483 - Random Forest n_estimators = 40 (labelencoder + escalonamento)

0.8223 - KNN (labelencoder + onehotencoder + escalonamento)
0.7746 - KNN (labelencoder)
0.7760 - KNN (labelencoder + onehotencoder)
0.8219 - KNN (labelencoder + escalonamento)

0.8495 - Regressão Logística (labelencoder + onehotencoder + escalonamento)
0.7910 - Regressão Logística (labelencoder)
0.7955 - Regressão Logística (labelencoder + onehotencoder)
0.8184 - Regressão Logística (labelencoder + escalonamento)

0.8507 - SVM Linear (labelencoder + onehotencoder + escalonamento)
?      - SVM Linear (labelencoder)
?      - SVM Linear (labelencoder + onehotencoder)
0.8184 - SVM Linear (labelencoder + escalonamento)

0.8233 - Rede Neural (labelencoder + onehotencoder + escalonamento)
0.2440 - Rede Neural (labelencoder)
0.7881 - Rede Neural (labelencoder + onehotencoder)
0.8481 - Rede Neural (labelencoder + escalonamento)