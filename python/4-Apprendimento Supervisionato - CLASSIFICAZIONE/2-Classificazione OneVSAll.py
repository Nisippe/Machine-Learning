import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.metrics import accuracy_score
%matplotlib inline


digits = load_digits()


X = digits.data
Y = digits.target

for i in range(0,10):
    pic_matrix = X[Y==i][0].reshape([8,8])
    # selezioniamo il primo esempio corrispondente alla classe corrente
    #ed utilizziamo reshape per ottenere una matrice 8x8 dal vettore
    plt.imshow(pic_matrix, cmap="gray")
    plt.show()

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=0)
X_train.shape


from sklearn.preprocessing import MinMaxScaler

mms = MinMaxScaler()
X_train = mms.fit_transform(X_train)
X_test = mms.transform(X_test)

lr = LogisticRegression()
lr.fit(X_train, Y_train)

y_pred_proba = lr.predict_proba(X_test)
y_pred = lr.predict(X_test)

print("ACCURACY: "+str(accuracy_score(Y_test, y_pred)))
print("LOG LOSS: "+str(log_loss(Y_test, y_pred_proba)))

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(Y_test,y_pred)
cm

import seaborn as sns
plt.figure(figsize=(9,9))
sns.heatmap(cm,annot=True,cmap="Blues_r",linewidths=.5,square=True)
plt.ylabel('Classe corretta')
plt.xlabel('Classe Predetta')



from sklearn.multiclass import OneVsRestClassifier

ovr = OneVsRestClassifier(LogisticRegression()) # Utilizziamo la regressione logistica come classificatore
ovr.fit(X_train, Y_train)

y_pred_proba = ovr.predict_proba(X_test)
y_pred = ovr.predict(X_test)

print("ACCURACY: "+str(accuracy_score(Y_test, y_pred)))
print("LOG LOSS: "+str(log_loss(Y_test, y_pred_proba)))
