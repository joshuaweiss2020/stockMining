import pandas as pd
import numpy as np
import seaborn as sns
help(sns)

planets = sns.load_dataset('planets')
print(planets.shape)
print(planets.dropna().describe())

df = pd.DataFrame(np.random.randint(1,10,(4,4)),columns=list("ABCD"))
print(df)
print(df["A"].mean())

print(df.loc[0].mean())
print(df.mean())
print(df.mean(axis=1))

print(planets.head())

gb_year = planets.groupby("year")
print(gb_year["distance"].count())
print(gb_year["number"].count())

s0 = pd.Series([2019,2019,2019,2020,2020])
s1 = pd.Series(np.linspace(1,10,5))
s2 = pd.Series(np.linspace(1,100,5))
df = pd.DataFrame({"A0":s0,"A1":s1,"A2":s2})
df.loc[0:0,"A1":"A1"]=np.nan
print(df)

print(df.groupby("A0").count())
print("_________________-")
for (A0,A12) in df.groupby("A0"):
    print("A0:{},A12 shape:{}".format(A0,A12.shape))

    # print("A1:{} A2:{}".format(A1,A2))