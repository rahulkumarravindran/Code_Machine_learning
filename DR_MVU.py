# -*- coding: utf-8 -*-
"""
Created on Thu Oct 21 16:25:41 2021

@author: Administrator
"""

import numpy as np
import pandas as pd
from MVU import MaximumVarianceUnfolding
import time


start=time.time()

colNames=[i for i in range(0,355)]
origData=pd.read_csv(r"D:\Windsor\Fourth semester\Applied Machine learning\Project\Datasets\Datasets\D1.csv",header=None,names=colNames)
m=len(origData)
n=len(colNames)
reducedData=origData[0:200][colNames[1:20]]
#reducedData[len(reducedData)+1]=origData[0:100][colNames[-1]]

MVUobject=MaximumVarianceUnfolding()
embeddedData=MVUobject.fit_transform(data=reducedData,dim=5,k=4)
end=time.time()
print(end-start)