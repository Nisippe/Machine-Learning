import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
iris = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data", names=["sepal length","sepal width","petal length","petal width","class"])


X=iris.drop("class",axis=1).values
Y=iris["class"].values

X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.3,random_state=0)

le=LabelEncoder()
Y_train=le.fit_transform(Y_train)
Y_test=le.transform(Y_test)

ss=StandardScaler()
X_train=ss.fit_transform(X_train)
X_test=ss.transform(X_test)

X2_train=X_train[:,:2]
X2_test=X_test[:,:2]

from sklearn.svm import LinearSVC

svc=LinearSVC()
svc.fit(X2_train,Y_train)
print("ACCURACY con 2 proprietà: Train=%.4f Test=%.4f" % (svc.score(X2_train, Y_train), svc.score(X2_test,Y_test)))

from viz import plot_bounds

plot_bounds((X2_train,X2_test),(Y_train,Y_test),svc,figsize=[12,8])

svc=LinearSVC()
svc.fit(X_train,Y_train)
print("ACCURACY  Train=%.4f Test=%.4f" % (svc.score(X_train, Y_train), svc.score(X_test,Y_test)))
