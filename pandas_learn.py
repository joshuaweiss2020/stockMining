import pandas as pd
import numpy as np

data = pd.Series(np.random.random((5)), index=["a", "b", "c", "d", "f"])

print(data)

data = pd.Series(np.linspace(0, 1, 5), index=["a", "b", "c", "d", "f"])

print(data)

dict1 = {"a": 1, "d": 3, "c": 5, "b": 0, "ee": 11}

data = pd.Series(dict1, index=["ee", "d", "b"])

print(data)
print(data[0], data[1], data[2])

print(data["ee":"b"])

print("data.index:", data.index)

data_col1 = pd.Series(dict1)
data_col2 = data_col1 + 1

df = pd.DataFrame({"col1": data_col1, "col2": data_col2})

print(df)
print("col1:")
print(df["col1"])
#

data = [{"a": i, "b": i + 1} for i in range(4)]
df = pd.DataFrame(data)
print(data)
print(df)

data = np.random.rand(2, 3)
df = pd.DataFrame(data, columns=["c1", "c2", "c3"], index=["r1", "r2"])

print(df)

data = np.ones(6, dtype=[("C1", 'i8'), ("C2", "f8")])

df = pd.DataFrame(data, index=["r" + str(i) for i in range(6)])

print(df)
####series 取值
data = pd.Series(np.linspace(0, 1, 5), index=["a", "b", "c", "d", "f"])
data1 = pd.Series(np.linspace(0, 1, 5))
print(data.loc["a":"c"])
print(data["a":"c"])
print("data1:")
print(data1[1:3])  # 隐式
print(data1.loc[1:3])
print(data1.iloc[1:3])  # 隐式

####dataFrame 取值
np.random.seed(9)
data = np.random.randint(1, 10, (4, 5))
df = pd.DataFrame(data, columns=["c" + str(i) for i in range(1, 6)], index=["r" + str(i) for i in range(1, 5)])

print(df)

print(df["c1"])

# df["sum"] =df["c5"] * 2
#
df["sum"] = pd.Series([v.sum() for v in df.values], index=df.index)
print("add sum")
print(df)

print(df.values[0])

print(df["r1":"r1"])

print(df.loc["r1":"r2", "c1":"c2"])

print(df.loc[df.c2 > 6, "c1":])

print(df.loc[:, df.values[0] > 7])

#####通用函数
s1 = pd.Series(np.linspace(0, 10, 5, dtype='int'), index=["a", "b", "c", "d", "f"])
s2 = pd.Series(np.linspace(0, 10, 3, dtype='int'), index=list("adf"))
print(s1, "\n", s2)
print(s1 + s2)
print(s1.add(s2, fill_value=0))

s1 = pd.Series(np.linspace(5, 10, 5, dtype='int'), index=list("abcde"))
s2 = pd.Series(np.linspace(0, 5, 5, dtype='int'), index=list("ebdca"))

df = pd.DataFrame({"s1": s1, "s2": s2})
print(df)

print(df - df.iloc[0])

print(df.subtract(df["s1"], axis=0))

# 缺失值
s1 = np.array([33, np.nan, 44, np.nan, 33])
s2 = pd.Series([33, np.nan, 44, np.nan, 33], index=list("abcde"))

print("s1:", s1)
print("s1 sum:", s1.sum())
print("s2 sum:", s2.sum())
print("s1 nansum:", np.nansum(s1))

print("s2 is null:", s2.isnull())

print("s2 is not null:", s2[s2.notnull()])
print("s2 dropna:", s2.dropna())

print("s2 fillna\n", s2.fillna(999), "\n", s2.ffill(), "\n", s2.bfill())

df = pd.DataFrame(np.random.randint(1, 10, (5, 5)), index=list("abcde"))

df.iloc[0:1, 1:2], df.iloc[2:3, 2:3], df.iloc[3:4, 1:2] = np.nan, np.nan, np.nan
# df.iloc[0:3,1:2]=np.nan
print(df)

print(df.dropna())

print(df.dropna(axis=1, how='all'))

print(df.dropna(axis="columns", thresh=4))

print(df.fillna(axis=0, method="ffill"))

print(df.fillna(value=999, axis=0))

# 多级索引
index = [("year", 2020), ("year", 2019), ("month", 1), ("month", 2), ("day", 10), ("day", 20)]
vals = [3, 4, 5, 6, 7, 8]

data = pd.Series(vals, index=index)
print("data:\n", data)
index = pd.MultiIndex.from_tuples(index)
data = data.reindex(index)
print("data:\n", data)
print(data[:, 2020])
df = data.unstack()
print(df)
print(df.stack())
print(data["year", 2020])

index = [("year1", 2020), ("year1", 2019), ("year2", 2020), ("year2", 2019), ("year3", 2019), ("year3", 2020)]
vals = [3, 4, 5, 6, 7, 8]
data = pd.Series(vals, index=index)
index = pd.MultiIndex.from_tuples(index)
data = data.reindex(index)
print("data:\n", data)
print("data:\n", data.unstack())

df = pd.DataFrame(np.random.randint(1, 10, (4, 4)), index=[['a', 'a', 'b', 'b'], [1, 2, 1, 2]], columns=list("ABCD"))
print(df)
data = {("year1", 2020):3, ("year1", 2019): 4, ("year2", 2020): 5, ("year2", 2019): 6, ("year3", 2019):7,(
"year3", 2020):7}
data = pd.Series(data)
print(data.unstack())
