# -*- coding: utf-8 -*-
"""
Created on Sat May 21 20:14:04 2022

@author: technOrbit
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

pd.options.display.max_columns=None

df=pd.read_csv("StudentPerformance.csv")
print(df)

print(df.notnull())
series = pd.notnull(df["math score"]) 
print(df[series])

print(df.isnull())
series = pd.isnull(df["math score"]) 
print(df[series])

from sklearn.preprocessing import LabelEncoder 
le = LabelEncoder()
df['gender'] = le.fit_transform(df['gender']) 
newdf=df
print(df)


#filling null values
m_v=df['math score'].mean()
df['math score'].fillna(value=m_v, inplace=True) 
print(df)

#replacing null values with NaN
missing_values = ["Na", "na"]
df = pd.read_csv("StudentPerformance.csv", na_values = missing_values)
print(df)


df.replace(to_replace = np.nan, value = -99)

#To drop rows with at least 1 null value
print(df.dropna())

#To Drop rows if all values in that row are missing
print(df.dropna(how = 'all'))

#To Drop columns with at least 1 null value.
print(df.dropna(axis = 1))

#Select the columns for boxplot and draw the boxplot.
col = ['math score', 'reading score' , 'writing score','placement score']
df.boxplot(col)

print(np.where(df['math score']>90))
print(np.where(df['reading score']<25)) 
print(np.where(df['writing score']<30))

fig, ax = plt.subplots(figsize = (18,10)) 
ax.scatter(df['placement score'], df['placement offer count'])
plt.show()

#Detecting outliers using Z-Score:
z = np.abs(stats.zscore(df['math score']))
print(z)

threshold = 0.18
sample_outliers = np.where(z <threshold)
print(sample_outliers)

#Detecting outliers using Inter Quantile Range(IQR):
sorted_rscore= sorted(df['reading score'])
print(sorted_rscore)
q1 = np.percentile(sorted_rscore, 25)
q3 = np.percentile(sorted_rscore, 75)
print(q1,q3)
IQR = q3-q1
lwr_bound = q1-(1.5*IQR)
upr_bound = q3+(1.5*IQR)
print(lwr_bound, upr_bound)
 
r_outliers = []
for i in sorted_rscore:
    if (i<lwr_bound or i>upr_bound):
        r_outliers.append(i)
print(r_outliers)

#Trimming/removing the outlier:
new_df=df
for i in sample_outliers:
    new_df.drop(i,inplace=True)
print(new_df)

df_stud=df
ninetieth_percentile = np.percentile(df_stud['math score'], 90)
b = np.where(df_stud['math score']>ninetieth_percentile,
ninetieth_percentile, df_stud['math score'])
print("New array:",b)
df_stud.insert(1,"m score",b,True)
print(df_stud)

col = ['reading score']
df.boxplot(col)

median=np.median(sorted_rscore)
print(median)
refined_df=df
refined_df['reading score'] = np.where(refined_df['reading score'] >upr_bound, median,refined_df['reading score'])

refined_df['reading score'] = np.where(refined_df['reading score'] <lwr_bound, median,refined_df['reading score'])
print(refined_df)

col = ['reading score']
refined_df.boxplot(col)

new_df['math score'].plot(kind = 'hist')

df['log_math'] = np.log10(df['math score'])

df['log_math'].plot(kind = 'hist')