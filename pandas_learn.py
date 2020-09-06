import pandas as pd
import numpy as np

data = pd.Series(np.random.random((5)),index=["a","b","c","d","f"])

print(data)

data = pd.Series(np.linspace(0,1,5),index=["a","b","c","d","f"])

print(data)

dict1 = {"a":1,"d":3,"c":5,"b":0,"ee":11}

data = pd.Series(dict1,index=["ee","d","b"])

print(data)
print(data[0],data[1],data[2])

print(data["ee":"b"])

print("data.index:",data.index)

data_col1 = pd.Series(dict1)
data_col2 = data_col1 + 1

df = pd.DataFrame({"col1":data_col1,"col2":data_col2})

print(df)
print("col1:")
print(df["col1"])
#

data =[{"a":i,"b":i+1} for i in range(4)]
df = pd.DataFrame(data)
print(data)
print(df)

data = np.random.rand(2,3)
df = pd.DataFrame(data,index=[1,2])

print(df)