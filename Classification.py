# -*- coding: utf-8 -*-
"""
Created on Fri Nov 12 19:30:13 2021

@author: Rahul Kumar
"""

#import needed library and modules
from sklearn.model_selection import train_test_split,cross_val_score,ShuffleSplit
from sklearn.preprocessing import StandardScaler
import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn import svm,tree
from sklearn.metrics import f1_score,accuracy_score,precision_score,confusion_matrix,recall_score
from time import time

def evaluate(y_pred,y_true,m_test):
    
    #Calculating the F1_score
    f1=f1_score(y_true,y_pred,average='weighted',zero_division=1)
    
    #Calculating the accuracy
    Accuracy=accuracy_score(y_true,y_pred)
    
    #Calculating the confusion matrix
    ConfMatrix=confusion_matrix(y_true, y_pred)
    
    #Calculating precision
    precision=precision_score(y_true,y_pred,average='macro',zero_division=1)
    
    #Calculating recall score
    recall=recall_score(y_true,y_pred,average='macro',zero_division=1)
    
    return {'f1_score': f1, 'Accuracy':Accuracy,'ConfusionMatrix': ConfMatrix,"Precision": precision,"Recall" : recall}
    

def NaiveBayes(x_train,x_test,y_train,y_test,NameOfDR):
    #Initializing the Naive bayes classifier object
    gnb=GaussianNB()

    #predicting the labels using the model trained by train data
    y_pred=gnb.fit(x_train,y_train).predict(x_test)

    #Finding the shape of test data
    (m_test,n_test)=x_test.shape
    
    #cross validation
    cv = ShuffleSplit(n_splits=10, test_size=0.1, random_state=0)
    cv_score=cross_val_score(gnb,x_train.append(x_test),y_train.append(y_test),cv=cv)

    #Calculating the accuracy of the model by comparing the predicted labels and the true test labels
    Accuracy=100-((y_pred!=y_test).sum()/m_test)
    
    #Evaluating the model for F1_score, Accuracy, precision, Recall and Confusion matrix
    eval_results=evaluate(y_pred, y_test, m_test)
    

    return print("Accuracy for the {} dataset and Naive Bayes Classifier:{}".format(NameOfDR,cv_score.mean()))

def SupportVectorMachine(x_train,x_test,y_train,y_test,NameOfDR):

    #Initializing the Support Vector Machine classifier object
    SVM=svm.SVC()

    #predicting the labels using the model trained by train data
    y_pred=SVM.fit(x_train,y_train).predict(x_test)

    #Finding the shape of test data
    (m_test,n_test)=x_test.shape
    
    #cross validation
    cv = ShuffleSplit(n_splits=4, test_size=0.1, random_state=0)
    cv_score=cross_val_score(SVM,x_train.append(x_test),y_train.append(y_test),cv=cv)

    #Calculating the accuracy of the model by comparing the predicted labels and the true test labels
    Accuracy=100-((y_pred!=y_test).sum()/m_test)
    
    #Evaluating the model for F1_score, Accuracy, precision, Recall and Confusion matrix
    eval_results=evaluate(y_pred, y_test, m_test)

    return print("Accuracy for the {} dataset and Support Vector Machine Classifier:{}".format(NameOfDR,cv_score.mean()))

def DecisionTree(x_train,x_test,y_train,y_test,NameOfDR):
    
    #Initializing the Decision Tree classifer object
    DT=tree.DecisionTreeClassifier()
    
    #predicting the labels using the model trained by train data
    y_pred=DT.fit(x_train.append(x_test),y_train.append(y_test)).predict(x_test)

    #Finding the shape of test data
    (m_test,n_test)=x_test.shape
    
    #cross validation
    cv = ShuffleSplit(n_splits=10, test_size=0.1, random_state=0)
    cv_score=cross_val_score(DT,x_train.append(x_test),y_train.append(y_test),cv=cv)

    #Calculating the accuracy of the model by comparing the predicted labels and the true test labels
    Accuracy=100-((y_pred!=y_test).sum()/m_test)
    
    #Evaluating the model for F1_score, Accuracy, precision, Recall and Confusion matrix
    eval_results=evaluate(y_pred, y_test, m_test)

    return print("Accuracy for the {} dataset and Decision Tree Classifier:{}".format(NameOfDR,cv_score.mean()))


tic=time()

#for easy viewing
for i in range(2):
    print()
print("-"*50)

#read the data from the original CSV 
data=pd.read_csv(r"D:\Windsor\Fourth semester\Applied Machine learning\Project\Datasets\Datasets\D1.csv",header=None)

#List of dimension reduced datasets
DR_datasets={'Conformal Eigenmaps':r"D:\Windsor\Fourth semester\Applied Machine learning\Project\Code_Machine_learning\DimensionReducedDataSet\D13_CE.csv",'Maximum Variance Unfolding':r"D:\Windsor\Fourth semester\Applied Machine learning\Project\Code_Machine_learning\DimensionReducedDataSet\D13_MVU.csv", "Landmark MVU": r"D:\Windsor\Fourth semester\Applied Machine learning\Project\Code_Machine_learning\DimensionReducedDataSet\D13_LMVU.csv"}

#Data preprocessing (Filling missing values and removing empty columns)
data=data.fillna(0)
data=data.loc[:,(data!=0).any(axis=0)]

#Finding the shape of the dataframe
(m,n)=data.shape

#Separating the data and the labels
x=data.loc[:,:n-2]
y=data.loc[:,n-1]

#Scale the data
scaler=StandardScaler()
#x=pd.Series(scaler.fit_transform(x).reshape(,1))

#The list of classifiers
listOfClassifiers={'Naive Bayes':NaiveBayes, 'Support Vector Machine':SupportVectorMachine, "Decision Tree":DecisionTree}

for i in listOfClassifiers:
    
    print("Running the {} classifier on Original and Dimesnion Reduced Datasets:".format(i))
    print()
    
    
    ##For the original Dataset
    #Splitting the data into test and train data
    x_train,x_test,y_train,y_test= train_test_split(x,y,test_size=0.3,random_state=39 )
    
    #Run the classifier on original dataset
    listOfClassifiers[i](x_train,x_test,y_train,y_test,'Original')
    
    #Iterating over the Dimension reduced Datasets
    for j in DR_datasets:
        
        #Reading the dimension Reduced dataset
        #The dataset is already preprocessed, So there is no need to fill missing values or Scale
        data_dr=pd.read_csv(DR_datasets[j],header=None)
        
        #find the shape of Dimesnion Reduced dataset
        (m_dr,n_dr)=data_dr.shape
        
        #Separating the data and the labels
        x_dr=data_dr.loc[:,:n_dr-2]
        y_dr=data_dr.loc[:,n_dr-1]
        #y_dr=y[1:m_dr+1]
        
        ##For the Dimension Reduced Dataset
        #Splitting the data into test and train data
        x_train,x_test,y_train,y_test= train_test_split(x_dr,y_dr,test_size=0.3,random_state=39 )
        
        #Run the classifier on the Dimesion reduced dataset.
        listOfClassifiers[i](x_train,x_test,y_train,y_test,j)
    
    print()    
    print('-'*40)
    print()
        
        
toc=time()
print("Time Taken: {} mins".format((toc-tic)/60))
    