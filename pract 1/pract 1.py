# -*- coding: utf-8 -*-
"""
Created on Sun May 22 13:16:29 2022

@author: technOrbit
"""

import pandas as pd
import numpy as np
from sklearn import preprocessing

col_names= ['Sepal_Length','Sepal_Width','Petal_Length','Petal_Width','Species']
df = pd.read_csv("iris.data",names=col_names)
#print(df.head())

#print(df.head(n=5))

#print(df.tail(n=5))

#print(df.index)

#print(df.columns)

print(df.shape)
"""
print(df.dtypes)

print(df.columns.values)

print(df.describe(include='all'))

print(df[col_names])

print(df.sort_index(axis=1,ascending=False))

print(df.sort_values(by=col_names))

print(df.iloc[5])

print(df[0:3])
"""
print(df.isnull())

print(df.isnull().any())

#print(df.isnull().sum().sum())

#print(df.isnull().sum(axis=1))

print(df.isnull().sum())

print(df['Species'].unique())

#Data normalization:
x = df[['Sepal_Length']].values.astype(float)
min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(x)
df_normalized = pd.DataFrame(x_scaled)
print(df_normalized)

print(df['Species'].unique())
label_encoder = preprocessing.LabelEncoder()
df['Species']= label_encoder.fit_transform(df['Species'])
print(df['Species'].unique())


one_hot_df=pd.get_dummies(df,prefix="Species",columns=['Species'],drop_first=False)
print(one_hot_df)

