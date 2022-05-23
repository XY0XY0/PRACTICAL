# -*- coding: utf-8 -*-
"""
Created on Sun May 22 12:51:26 2022

@author: technOrbit
"""

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
df=pd.read_csv('iris.data',header=None)
df.columns = ["col1",'col2',"col3","col4","col5"]
print(df.head())

column=len(list(df))
print(column)

print(df.info())

print(np.unique(df["col5"]))


#histogram
fig, axes = plt.subplots(2,2,figsize=(24,12))

axes[0,0].set_title("Distribution of First column")
axes[0,0].hist(df["col1"])

axes[0,1].set_title("Distribution of Second column")
axes[0,1].hist(df["col2"])

axes[1,0].set_title("Distribution of Third column")
axes[1,0].hist(df["col3"])

axes[1,1].set_title("Distribution of Fourth column")
axes[1,1].hist(df["col4"])

plt.show()

#boxplot

data_to_plot=[df["col1"],df["col2"],df["col3"],df["col4"]]
sns.set_style("whitegrid")

#creating a figure instance
fig = plt.figure(1,figsize=(12,8))

#creating an axes instances
ax=fig.add_subplot(111)

#creating the boxplot
bp=ax.boxplot(data_to_plot)

print(df.describe())
