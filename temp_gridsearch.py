# -*- coding: utf-8 -*-
"""
Created on Mon Nov 22 23:33:14 2021

@author: Rahul
"""

from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.svm import SVC
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler,OneHotEncoder


data=pd.read_csv(r"D:\Windsor\Fourth semester\Applied Machine learning\Project\Datasets\Datasets\D13.csv",header=None)

#Data preprocessing (Filling missing values and removing empty columns)
data=data.fillna(0)
data=data.loc[:,(data!=0).any(axis=0)]

#Finding the shape of the dataframe
(m,n)=data.shape
print(data.shape)
#Separating the data and the labels
x=data.iloc[:,:-1]
y=data.iloc[:,-1:]
print(n)
print(x.shape)

parameters = {'kernel':('linear', 'rbf'), 'C':[1, 10,20]}
svc = SVC(gamma='auto')
clf = GridSearchCV(svc, parameters,cv=5,return_train_score=False)
clf.fit(x,y.values.ravel())
results=pd.DataFrame(clf.cv_results_)
print(results)

#One hot encoder
"""enc=OneHotEncoder(sparse=False)
y=enc.fit_transform(y)
print(y.shape)

"""