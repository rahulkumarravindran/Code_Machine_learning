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
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler,OneHotEncoder

data=pd.read_csv(r"D:\Windsor\Fourth semester\Applied Machine learning\Project\Datasets\Datasets\D13.csv",header=None)

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

n_y=len(list(set(y)))


#Scale the data

#print(y)

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=40)

x_train=tf.stack(x_train)
y_train=tf.stack(y_train)
x_test=tf.stack(x_test)
y_test=tf.stack(y_test)

#(m,n)= x_train.shape

model.add(LSTM(128,input_shape=(1,n-2),activation='relu',return_sequences=True))
#model.add(Dropout(0.2))
model.add(Dense(128,activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(32,activation='relu'))
#model.add(Dropout(0.2))
#no of classes in output
model.add(Dense(n_y+1,activation='softmax'))

opt=tf.keras.optimizers.Adam(learning_rate=1e-3,decay=1e-5)

model.compile(loss='sparse_categorical_crossentropy',optimizer='adam',metrics=['accuracy'])

model.fit(x_train,y_train,epochs=10,validation_data=(x_test,y_test))
y_pred=model.predict(x_test)
"""print(y_test)
accuracy =list(y_pred!=y_test).sum()/len(y_test)
print(accuracy)
"""