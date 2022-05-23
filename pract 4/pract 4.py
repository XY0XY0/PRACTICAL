# -*- coding: utf-8 -*-
"""
Created on Sat May 21 23:40:39 2022

@author: technOrbit
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

#%matplotlib inline

from sklearn.datasets import load_boston
boston_dataset = load_boston()

#print (boston_dataset.keys())

print(boston_dataset.DESCR)

data = pd.DataFrame(boston_dataset.data, columns=boston_dataset.feature_names)
print(data.head())

data['Price'] = boston_dataset.target
print(data.head())

print(data.describe())

print(data.info())

#Data preprocessing
print(data.isnull().sum())

sns.set(rc={'figure.figsize':(11.7,8.27)})
sns.displot(data['Price'],bins=30);
plt.show()

correlation_matrix=data.corr().round(2)
sns.heatmap(data=correlation_matrix,annot=True)

plt.figure(figsize=(20,5))
features = ['LSTAT','RM']
target = data['Price']

for i, col in enumerate(features):
    plt.subplot(1,len(features),i+1)
    x=data[col]
    y=target
    plt.scatter(x,y,marker='o')
    plt.title(col)
    plt.xlabel(col)
    plt.ylabel('MEDV')
    
#training and Testing the model
X = pd.DataFrame(np.c_[data['LSTAT'], data['RM']], columns= ['LSTAT', 'RM'])
Y= data['Price']

from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test=train_test_split(X, Y, test_size = 0.2, random_state=5)
print(X_train.shape) 
print(X_test.shape)
print(Y_train.shape)
print(Y_test.shape)

from sklearn.linear_model import LinearRegression

model =LinearRegression() 
print(model.fit(X_train, Y_train))

#model evalution
from sklearn.metrics import mean_squared_error

from sklearn.metrics import r2_score
y_pred= model.predict(X_test)

rmse=(np.sqrt(mean_squared_error(Y_test,y_pred))) 
r2= r2_score(Y_test,y_pred)

print("The Model performance for Testing set")
print(".............................................")
print( 'RMSE is {}'.format(rmse))
print('R2 score is {}'.format(r2))

#predicting selling price
sample_data= [[6.89, 9.939]]

price=model.predict(sample_data)
print("predicted selling price for house: ${:,.2f}".format(price[0]))
