import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


boston = pd.read_csv("/home/giuseppe/Scrivania/Programming/CSV/housing.data.csv", sep='\s+',names=["CRIM","ZN","INDUS","CHAS","NOX","RM","AGE","DIS","RAD","TAX","PRATIO","B","LSTAT","MEDV"])
boston.head()

boston.corr()


import seaborn as sns

hm = sns.heatmap(boston.corr(),xticklabels=boston.columns,yticklabels=boston.columns)

cols=['RM',"ZN","LSTAT","PRATIO","TAX","INDUS","MEDV"]

hmcols=sns.heatmap(boston[cols].corr(),xticklabels=boston[cols].columns,yticklabels=boston[cols].columns,
    annot=True,annot_kws={'size':13})

sns.pairplot(boston[cols])

X=boston[['RM','LSTAT']].values
Y=boston['MEDV'].values

X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.3, random_state=0)

ll=LinearRegression()
ll.fit(X_train,Y_train)
Y_pred=ll.predict(X_test)

print("MSE: "+str(mean_squared_error(Y_test,Y_pred)))
print("R2: "+str(r2_score(Y_test,Y_pred)))






X=boston.drop("MEDV",axis=1).values
Y=boston["MEDV"].values

X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.3, random_state=0)


from sklearn.preprocessing import StandardScaler

ss = StandardScaler()
X_train_std = ss.fit_transform(X_train)
X_test_std = ss.transform(X_test)

ll=LinearRegression()
ll.fit(X_train_std,Y_train)
Y_pred=ll.predict(X_test_std)

print("MSE: "+str(mean_squared_error(Y_test,Y_pred)))
print("R2: "+str(r2_score(Y_test,Y_pred)))
weights = pd.DataFrame(data=list(zip(boston.columns, ll.coef_)), columns=['feature', 'weight'])
weights
 
