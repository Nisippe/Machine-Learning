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
ss = StandardScaler()
X_train_poly = ss.fit_transform(X_train_poly)
X_test_poly = ss.transform(X_test_poly)

def overfit_eval(model, X, Y):

    """
    model: il nostro modello predittivo già addestrato
    X: una tupla contenente le prorietà del train set e test set (X_train, X_test)
    Y: una tupla contenente target del train set e test set (Y_train, Y_test)
    """

    Y_pred_train = model.predict(X[0])
    Y_pred_test = model.predict(X[1])

    mse_train = mean_squared_error(Y[0], Y_pred_train)
    mse_test = mean_squared_error(Y[1], Y_pred_test)

    r2_train = r2_score(Y[0], Y_pred_train)
    r2_test = r2_score(Y[1], Y_pred_test)

    print("Train set:  MSE="+str(mse_train)+" R2="+str(r2_train))
    print("Test set:  MSE="+str(mse_test)+" R2="+str(r2_test))

ll = LinearRegression()
ll.fit(X_train_poly, Y_train)

overfit_eval(ll, (X_train_poly, X_test_poly),(Y_train, Y_test))



from sklearn.linear_model import Ridge

alphas = [0.0001, 0.001, 0.01, 0.1 ,1 ,10] #alpha corrispone a lambda

for alpha in alphas:
    print("Alpha="+str(alpha))
    ridge = Ridge(alpha=alpha)
    ridge.fit(X_train_poly, Y_train)

    overfit_eval(ridge, (X_train_poly, X_test_poly),(Y_train, Y_test))


from sklearn.linear_model import Lasso

for alpha in alphas:
    print("Alpha="+str(alpha))
    lasso = Lasso(alpha=alpha)
    lasso.fit(X_train_poly, Y_train)

    overfit_eval(lasso, (X_train_poly, X_test_poly),(Y_train, Y_test))


from sklearn.linear_model import ElasticNet

for alpha in alphas:
    print("Lambda is: "+str(alpha))
    elastic = ElasticNet(alpha=alpha, l1_ratio=0.5)
    elastic.fit(X_train_poly, Y_train)
    overfit_eval(elastic, (X_train_poly, X_test_poly),(Y_train, Y_test))
