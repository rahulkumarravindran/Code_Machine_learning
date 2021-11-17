# -*- coding: utf-8 -*-
"""
Created on Wed Nov 10 14:44:03 2021

@author: Rahul
"""

#import needed library and modules
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.naive_bayes import GaussianNB

#read the data from the original CSV 
data=pd.read_csv(r"D:\Windsor\Fourth semester\Applied Machine learning\Project\Datasets\Datasets\D13.csv",header=None)

#read the data from the dimension reduced dataset
data_dr=pd.read_csv(r"D:\Windsor\Fourth semester\Applied Machine learning\Project\Code_Machine_learning\DimensionReducedDataSet\D13_MVU.csv",header=None)

#Data preprocessing (Filling missing values and removing empty columns)
data=data.fillna(0)
data=data.loc[:,(data!=0).any(axis=0)]

#Finding the shape of the dataframe
(m,n)=data.shape
(m_dr,n_dr)= data_dr.shape

#Separating the data and the labels
x=data.loc[:,:n-1]
y=data.loc[:,n]


#Separating the data and the labels for dimension reduced dataset
x_dr=data_dr
y_dr=y[0:m_dr]

#Updating the DR dataset with labels
data_dr[:][str(n_dr+1)]=y[0:m_dr]

#For normal dataset
#Splitting the data into test and train data
x_train,x_test,y_train,y_test= train_test_split(x,y,test_size=0.2,random_state=39 )

#Initializing the Naive bayes classifier object
gnb=GaussianNB()

#predicting the labels using the model trained by train data
y_pred=gnb.fit(x_train,y_train).predict(x_test)

#Finding the shape of test data
(m_test,n_test)=x_test.shape

#Calculating the accuracy of the model by comparing the predicted labels and the true test labels
Accuracy=100-((y_pred!=y_test).sum()/m_test)

print("Accuracy for original dataset:{}".format(Accuracy))

##For the Dimension reduced Dataset
#Splitting the data into test and train data
x_train,x_test,y_train,y_test= train_test_split(x_dr,y_dr,test_size=0.2,random_state=39 )

#Initializing the Naive bayes classifier object
gnb=GaussianNB()

#predicting the labels using the model trained by train data
y_pred=gnb.fit(x_train,y_train).predict(x_test)

#Finding the shape of test data
(m_test,n_test)=x_test.shape

#Calculating the accuracy of the model by comparing the predicted labels and the true test labels
Accuracy=100-((y_pred!=y_test).sum()/m_test)

print("Accuracy for Dimesion reduced dataset:{}".format(Accuracy))



 
 