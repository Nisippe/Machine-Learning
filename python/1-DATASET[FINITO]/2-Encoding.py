import pandas as pd
import numpy as np

shirts = pd.read_csv("https://raw.githubusercontent.com/ProfAI/ml00/master/2%20-%20Datasets%20e%20data%20preprocessing/data/shirts.csv",index_col=0)
shirts.head()
x=shirts.values
x[:10]

#Con Pandas
dizionario={"S":0,"M":1,"L":2,"XL":3}

shirts["taglia"]=shirts["taglia"].map(dizionario)
shirts=pd.get_dummies(shirts,columns=["colore"])
shirts.head()




#Con numpy
fmap=np.vectorize(lambda f:dizionario[f])
x[:,0]=fmap(x[:,0])
x[:5]



#sklearn
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder


le=LabelEncoder()
enc=OneHotEncoder(categorical_features=[1])
x[:,1]=le.fit_transform(x[:,1])
x[:5]
x_parse=enc.
