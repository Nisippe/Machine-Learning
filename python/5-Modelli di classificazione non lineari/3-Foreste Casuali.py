import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss
from sklearn.metrics import accuracy_score

titanic = pd.read_csv("http://web.stanford.edu/class/archive/cs/cs109/cs109.1166/stuff/titanic.csv")
titanic.info()
titanic.head()
titanic = titanic.drop("Name",axis=1)
titanic = pd.get_dummies(titanic)

X=titanic.drop("Survived",axis=1).values
Y=titanic["Survived"].values

X_train , X_test , Y_train , Y_test = train_test_split(X,Y,test_size=0.3,random_state=0)

from sklearn.tree import DecisionTreeClassifier

tree=DecisionTreeClassifier(criterion="gini")
tree.fit(X_train,Y_train)

Y_pred_train=tree.predict(X_train)
Y_pred=tree.predict(X_test)


accuracy_train=accuracy_score(Y_train,Y_pred_train)
accuracy_test=accuracy_score(Y_test,Y_pred)

print("ACCURACY: TRAIN=%.4f TEST=%.4f" % (accuracy_train,accuracy_test))


#Essendo che soffre di overfitting
#Limitiamo la profondità dell'albero

tree = DecisionTreeClassifier(criterion="gini",max_depth=6)
tree.fit(X_train, Y_train)

y_pred_train = tree.predict(X_train)
y_pred = tree.predict(X_test)

accuracy_train = accuracy_score(Y_train, y_pred_train)
accuracy_test = accuracy_score(Y_test, y_pred)

print("ACCURACY: TRAIN=%.4f TEST=%.4f" % (accuracy_train,accuracy_test))



#COMINCIA QUA
from sklearn.ensemble import RandomForestClassifier
forest=RandomForestClassifier(random_state=False,max_depth=8,n_estimators=30)
forest.fit(X_train,Y_train)
Y_pred_train=forest.predict(X_train)
Y_pred=forest.predict(X_test)

accuracy_train=accuracy_score(Y_train,Y_pred_train)
accuracy_test=accuracy_score(Y_test,Y_pred)
print("ACCURACY: TRAIN=%.4f TEST=%.4f" % (accuracy_train,accuracy_test))
