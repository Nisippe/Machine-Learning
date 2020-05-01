import pandas as pd

#Iris è un dataframe, struttura dati
iris = pd.read_csv("/home/giuseppe/Scrivania/Programming/CSV/iris.csv")
#Senza la prima riga del titolo pd.read_csv("path",header=None,names=[""])
iris.head()
iris.head(10)
iris.tail()
iris.tail(10)

iris.columns
iris.info()

#Altre cose base
