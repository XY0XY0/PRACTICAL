# -*- coding: utf-8 -*-
"""
Created on Sat May 21 23:20:04 2022

@author: technOrbit
"""

import pandas as pd

df = pd.read_csv("income.csv")
print(df.head())

print(df.age_group.unique())

#Using groupby()
print(df.groupby(df.age_group).count())

print(df.groupby(df.age_group).min())

print(df.groupby(df.age_group).max())

print(df.groupby(df.age_group).mean())

print(df.groupby(df.age_group).std())


#.describe() method
print(df.groupby(df.age_group).describe())

from sklearn import datasets 
data = datasets.load_iris() 
df = pd.DataFrame(data.data,columns=data.feature_names) 
df['species'] = pd.Series(data.target) 
print(df.head())

print(df.species.unique())

print(df.groupby(df.species))

print(df.groupby(df.species).count())

print(df.groupby(df.species).max())

print(df.groupby(df.species).min())

print(df.groupby(df.species).mean())

print(df.groupby(df.species).std())

print(df.groupby(df.species)["sepal length (cm)"].describe())

print(df.groupby(df.species)["sepal width (cm)"].describe())

print(df.groupby(df.species)["petal length (cm)"].describe())

print(df.groupby(df.species)["petal width (cm)"].describe())