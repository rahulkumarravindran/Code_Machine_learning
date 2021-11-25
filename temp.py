# -*- coding: utf-8 -*-
"""
Created on Sun Nov 21 21:04:45 2021

@author: Rahul
"""

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM,SimpleRNNCell
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split,cross_val_score,ShuffleSplit,KFold,StratifiedKFold,cross_validate
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils

def baseline_model(n):
	# create model
	model = Sequential()
	model.add(Dense(8, input_dim=n, activation='relu'))
	model.add(Dense(n, activation='softmax'))
	# Compile model
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	return model

def evaluate(y_pred,y_true):
    
    #Calculating the F1_score
    f1=f1_score(y_true,y_pred,average='weighted',zero_division=1)
    
    #Calculating the accuracy
    Accuracy=accuracy_score(y_true,y_pred)
    
    #Calculating the confusion matrix
    #ConfMatrix=confusion_matrix(y_true, y_pred)
    
    #Calculating precision
    precision=precision_score(y_true,y_pred,average='macro',zero_division=1)
    
    #Calculating recall score
    recall=recall_score(y_true,y_pred,average='macro',zero_division=1)
    
    return {'f1_score': f1, 'Accuracy':Accuracy,"Precision": precision,"Recall" : recall}
    

data=pd.read_csv(r"D:\Windsor\Fourth semester\Applied Machine learning\Project\Code_Machine_learning\DimensionReducedDataSet\D1_CE.csv",header=None)

#(m,n)=data.shape

#Data preprocessing (Filling missing values and removing empty columns)
data=data.fillna(0)
data=data.loc[:,(data!=0).any(axis=0)]

#Finding the shape of the dataframe
(m,n)=data.shape

model=Sequential()
x=list()
temp=data.loc[:,0:n-2]
scaler=StandardScaler()
temp=scaler.fit_transform(temp)
for i in temp:
    x.append([i])

y=list(map(int,data.loc[:,n-1]))

dummy_y = np_utils.to_categorical(y)

n_y=max(list(set(y)))

enc=OneHotEncoder(sparse=False)
y=enc.fit_transform(np.array(y).reshape(-1,1))

#Scale the data

print(len(y))
print(len(x))

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=40)

x_train=tf.stack(x_train)
y_train=tf.stack(y_train)
x_test=tf.stack(x_test)
y_test=tf.stack(y_test)

#(m,n)= x_train.shape

model.add(LSTM(100,input_dim=n-1,activation='relu',return_sequences=True))
model.add(LSTM(50,activation='relu',return_sequences=True))
#model.add(LSTM(50,activation='relu',return_sequences=True))
#model.add(LSTM(50,activation='relu',return_sequences=True))
model.add(Dense(n_y+1,activation='softmax'))

opt=tf.keras.optimizers.Adam(learning_rate=1e-3,decay=1e-5)

model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])

model.fit(x_train,y_train,epochs=10,validation_data=(x_test,y_test))

"""scoring = {'accuracy' : make_scorer(accuracy_score), 
           'precision' : make_scorer(precision_score),
           'recall' : make_scorer(recall_score), 
           'f1_score' : make_scorer(f1_score)}

cv = KFold(n_splits=10, shuffle=True, random_state=0)
result=cross_validate(model,x_train,y_train,cv=cv,score=make_scorer(accuracy_score))

#print(cv_score['accuracy'])

y_pred=model.predict(x_test)
print(y_test)
accuracy =list(y_pred!=y_test).sum()/len(y_test)
print(accuracy)


#result=evaluate(y_pred,y_test)

print(result.mean())


print(len(x))
print(len(y))
estimator = KerasClassifier(build_fn=baseline_model(n_y+1), epochs=200, batch_size=5, verbose=0)
kfold = KFold(n_splits=10, shuffle=True)
results = cross_val_score(estimator, x, dummy_y, cv=kfold)
print("Baseline: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))"""