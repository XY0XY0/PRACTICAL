# -*- coding: utf-8 -*-
"""
Created on Sun May 22 01:51:05 2022

@author: technOrbit
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
dataset = sns.load_dataset('titanic')
print(dataset.head())

#Ploting Histogram Matplotlib
#I.The Dist Plot
sns.distplot(dataset['fare'])

sns.distplot(dataset['fare'], kde=False)

sns.distplot(dataset['fare'], kde=False, bins=10)

#II. The Joint Plot :
sns.jointplot(x='age', y='fare', data=dataset)

sns.jointplot(x='age', y='fare', data=dataset, kind='hex')

dataset = dataset.dropna()

#III. The Pair Plot :
sns.pairplot(dataset)



#sns.pairplot(dataset, hue='sex')

#IV. The Rug Plot
print(sns.rugplot(dataset['fare']))
