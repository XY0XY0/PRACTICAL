# -*- coding: utf-8 -*-
"""
Created on Sun May 22 02:00:30 2022

@author: technOrbit
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

df=pd.read_csv('titanic-train.csv')
print(df.head())

cols = df.columns
print(cols)

print(df.info())

print(df.describe())

print(df.isnull().sum())

df.dropna()
print(df.isnull().sum()) 

sns.boxplot(df['Sex'],df['Age'])
plt.show()

sns.boxplot(df['Sex'],df['Age'],df['Survived'])
plt.show()