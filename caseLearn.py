import pandas as pd
import numpy as np

pop = pd.read_csv("data\state-population.csv")
areas = pd.read_csv("data\state-areas.csv")
abbrevs = pd.read_csv("data\state-abbrevs.csv")

print(pop.head(1));
print(areas.head(1));
print(abbrevs.head(1))

pop_abb = pd.merge(pop, abbrevs, how='outer',
                   left_on='state/region', right_on='abbreviation').drop('abbreviation',axis=1)

print(pop_abb.head())

print(pop_abb.isnull().any())

print(pop_abb[pop_abb["population"].isnull()])
print(pop_abb[pop_abb["state"].isnull()])

pop_abb.loc[pop_abb["state/region"]=="PR","state"] = "Puerto"
pop_abb.loc[pop_abb["state/region"]=="USA","state"] = "Unit States"
# pop_abb[pop_abb["state/region"]=="PR"]["state"] = 'Puerto'
print(pop_abb[pop_abb["population"].isnull()])

pop_abb_area = pd.merge(pop_abb,areas,how='outer')

print(pop_abb_area.isnull().any())

print(pop_abb_area[pop_abb_area["area (sq. mi)"].isnull()])

pop_abb_area.dropna(inplace=True)
print(pop_abb_area.isnull().any())
print(pop_abb_area.head())

pop_abb_area["density"] = pop_abb_area["population"]/pop_abb_area["area (sq. mi)"]
final = pop_abb_area.query("ages=='total' & year==2010")
final.set_index("state",inplace=True)
density = final["population"]/final["area (sq. mi)"]
density.sort_values(ascending=False,inplace=True)
print(density.head())
print(density.tail())