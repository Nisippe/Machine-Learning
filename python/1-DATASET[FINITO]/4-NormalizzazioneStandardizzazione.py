import pandas as pd
import numpy as np

iris = pd.read_csv("/home/giuseppe/Scrivania/Programming/CSV/iris.csv",usecols=[0,1,2])
iris.head()
features=["sepal.length","sepal.width","petal.length"]
to_norm=iris[features]
iris_nor[features]=(to_norm - to_norm.min())/(to_norm.max() - to_norm.min())
#Normalizzazione
from sklearn.preprocessing import MinMaxScaler
mms = MinMaxScaler()
X_norm = mms.fit_transform(iris)
X_norm[:5]
X_norm[features] = (to_norm-to_norm.min())/(to_norm.max()-to_norm.min())
wines_norm.head()

#Standardizzazione
to_std=iris.copy()[features]
iris_std=(to_std - to_std.mean())/to_std.std()

iris.head()
iris_std.head()



from sklearn.preprocessing import StandardScaler

ss = StandardScaler()
X = iris.values
X_std = ss.fit_transform(X)
X_std[:5]
