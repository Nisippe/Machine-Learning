from sklearn.preprocessing import *
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer

impo=SimpleImputer(missing_values=np.NaN,strategy="most_frequent")
iris = pd.read_csv("/home/giuseppe/Scrivania/Programming/CSV/iris.csv")
X=iris.values
impo.fit_transform(X)
print(X)
