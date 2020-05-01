#Librerie
import pandas as pd
from pandas.plotting import scatter_matrix
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import PolynomialFeatures
from sklearn.impute import SimpleImputer
import numpy as np
from sklearn.metrics import log_loss
from sklearn.metrics import accuracy_score
#Dataset principale
impo=SimpleImputer(missing_values=np.NaN,strategy="most_frequent")

titanic_test=pd.read_csv("/home/giuseppe/Scrivania/Programming/CSV/Titanic/test.csv")
titanic=pd.read_csv("/home/giuseppe/Scrivania/Programming/CSV/Titanic/train.csv")
titanic=pd.get_dummies(titanic,columns=['Sex'])
titanic=pd.get_dummies(titanic,columns=['Embarked'])
titanic_test=pd.get_dummies(titanic_test,columns=['Sex'])
titanic_test=pd.get_dummies(titanic_test,columns=['Embarked'])

titanic.shape
titanic.head()
titanic.describe()
titanic.groupby('Survived').size()
dataset.plot(kind='box',subplots=True,layout(2,4),sharex=False,sharey=False
scatter_matrix(titanic)

X=titanic.drop(['Name','Ticket','Cabin','Survived'],axis=1)
Y=titanic['Survived']
X_test=titanic_test.drop(['Name','Ticket','Cabin'],axis=1)
X=impo.fit_transform(X)
X_test=impo.fit_transform(X_test)
X_test_values=titanic_test['Name'].values

from sklearn.preprocessing import MinMaxScaler
mms=MinMaxScaler()
X=mms.fit_transform(X)
X_test=mms.transform(X_test)


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=0)
lr=LogisticRegression()
lr.fit(X,Y)
y_pred_proba=lr.predict_proba(X_test)
y_pred=lr.predict(X_test)

print("ACCURACY: "+str(accuracy_score(Y_test, y_pred)))
print("LOG LOSS: "+str(log_loss(Y_test, y_pred_proba)))


from sklearn.linear_model import LogisticRegression

lr=LogisticRegression()
lr.fit(X,Y)
y_pred_proba=lr.predict_proba(X_test)
y_pred=lr.predict(X_test)

for i in range(0,len(y_pred)):
    print(X_test_values[i]+" "+str(y_pred[i]))
