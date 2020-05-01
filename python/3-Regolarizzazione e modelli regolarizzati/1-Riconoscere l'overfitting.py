import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

boston = pd.read_csv("/home/giuseppe/Scrivania/Programming/CSV/housing.data.csv", sep='\s+',names=["CRIM","ZN","INDUS","CHAS","NOX","RM","AGE","DIS","RAD","TAX","PRATIO","B","LSTAT","MEDV"])
boston.head()
X = boston.drop('MEDV',axis=1).values
Y = boston['MEDV'].values

X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.3, random_state=0)

polyfeats = PolynomialFeatures(degree=2)
X_train_poly = polyfeats.fit_transform(X_train)
X_test_poly = polyfeats.transform(X_test)

print("Numero di esempi nel test di addestramento: "+str(X_train_poly.shape[0]))
print("Numero di features: "+str(X_train_poly.shape[1]))

ss=StandardScaler()
X_train_poly=ss.fit_transform(X_train_poly)
X_test_poly=ss.transform(X_test_poly)

ll=LinearRegression()
ll.fit(X_train_poly,Y_train)

Y_pred=ll.predict(X_train_poly)
mse = mean_squared_error(Y_pred,Y_train)
r2 = r2_score(Y_pred,Y_train)
print(str(mse)+","+str(r2))


Y_pred_test=ll.predict(X_test_poly)
mse_test = mean_squared_error(Y_pred_test,Y_test)
r2_test = r2_score(Y_pred_test,Y_test)
print(str(mse_test)+","+str(r2_test))
re
