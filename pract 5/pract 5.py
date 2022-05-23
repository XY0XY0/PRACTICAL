# -*- coding: utf-8 -*-
"""
Created on Sun May 22 00:27:04 2022

@author: technOrbit
"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from math import exp

data = pd.read_csv("data.csv")
print(data.head())

print(data.describe())


# Visualizing the dataset
plt.scatter(data['Age'], data['Purchased'])
plt.xlabel("age")
plt.ylabel("purchased")
plt.show()

# Divide the data to training set and test set
X_train, X_test, y_train, y_test = train_test_split(data['Age'], data['Purchased'],test_size=0.20)

#making prediction using scikit learn
from sklearn.linear_model import LogisticRegression

model=LogisticRegression()

model.fit(X_train.values.reshape(-1, 1), y_train.values.reshape(-1, 1).ravel())

y_pred_sk = model.predict(X_test.values.reshape(-1, 1))
plt.clf()
plt.scatter(X_test,y_test)
plt.scatter(X_test, y_pred_sk, c="red")

plt.xlabel("age")
plt.ylabel("purchased")

#Accuracy
print("f Accuracy = {lr_model.score(X_test.values.reshape(-1, 1), y_test.values.reshape(-1, 1))}")


#Confusion matrix
from sklearn.metrics import confusion_matrix

#extracting true_positives, false_positives, true_negatives, false_negatives
tn, fp, fn, tp = confusion_matrix(y_test, y_pred_sk).ravel()
print("True Negatives: ",tn)
print("False Positives: ",fp)
print("False Negatives: ",fn)
print("True Positives: ",tp)

#Accuracy
Accuracy = (tn+tp)*100/(tp+tn+fp+fn)
print("Accuracy {:0.2f}%:".format(Accuracy))

#Precision
Precision = tp/(tp+fp)
print("Precision {:0.2f}".format(Precision))

#Recall
Recall = tp/(tp+fn)
print("Recall {:0.2f}".format(Recall))

#Error rate
err = (fp + fn)/(tp + tn + fn + fp)
print("Error rate {:0.2f}".format(err))