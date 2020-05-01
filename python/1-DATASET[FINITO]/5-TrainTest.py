import pandas as pd
import numpy as np

from sklearn.datasets import load_boston
boston = load_boston()

X = boston.data
Y = boston.target

from sklearn.model_selection import train_test_split

X = boston.data
Y = boston.target

X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.3)

X_train.shape
X_test.shape

boston_df = pd.DataFrame(data= np.c_[boston['data'], boston['target']], columns= np.append(boston['feature_names'], 'TARGET'))
boston_df.head()

boston_test_df = boston_df.sample(frac=0.3)
boston_train_df = boston_df.drop(boston_test_df.index)

print("Numero di esempi nel Train set: "+str(boston_train_df.shape[0]))
print("Numero di esempi nel Test set: "+str(boston_test_df.shape[0]))
