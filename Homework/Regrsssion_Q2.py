# -*- coding: utf-8 -*-
"""
Created on Sat Oct 24 22:06:45 2020

@author: user
"""
import pandas as pd
import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
#%%
data = pd.read_csv(r"C:\Users\user\Desktop\NTU\Fintech\adult.data")
df = pd.get_dummies(data)  #one-hot encoding
data_set = np.array(df)
training_set = np.empty([int(data_set.shape[0]*0.8), data_set.shape[1]])
test_set = np.empty([int(data_set.shape[0]*0.2), data_set.shape[1]])
m=0 ;n= 0
for idx in range(training_set.shape[0]):
    training_set[idx]=data_set[idx]

for idx in range(test_set.shape[0]):
    test_set[idx]=data_set[idx]

mean = training_set.mean(axis=0) ;std = training_set.std(axis=0)
training_set_NM = (training_set-mean)/std
test_set_NM = (test_set-mean)/std

y_train_NM = training_set_NM[:,-1]
y_test_NM =  test_set_NM[:,-1]
y_train = training_set[:,-1]
y_test =  test_set[:,-1]
#%%
# =============================================================================
# linear regression
# =============================================================================
class Linear_Regression:

   def get_weight(self, X, Y):
      x_shape = X.shape
      X_b= np.concatenate([np.ones((x_shape[0],1)), X], axis=1)
      self.weight_matrix = np.matmul(np.matmul(np.linalg.pinv(np.matmul(X_b.T,X_b)),X_b.T), Y)                                                                                           
      return self.weight_matrix 
 
   def predict(self, X, weight):
      x_shape = X.shape
      X_b= np.concatenate([np.ones((x_shape[0],1)), X], axis=1)
      prediction = np.matmul(X_b, weight ) 
      return prediction
  
class Linear_Regression_nobias:
 
  def get_weight(self, X, Y):
      self.weight_matrix = np.matmul(np.matmul(np.linalg.pinv(np.matmul(X.T,X)),X.T), Y)                                                                                           
      return self.weight_matrix 
 
  def predict(self, X, weight):
      prediction = np.matmul(X, weight) 
      return prediction
  #%%
def rmse(x, y):
    return np.sqrt(((x - y) ** 2).mean())
def mse(x, y):
    return ((x - y) ** 2).mean()
#%%
x = np.delete(training_set_NM, 109, axis=1) 
x = np.delete(training_set_NM, 108, axis=1) 
y = np.delete(test_set_NM, 109, axis=1) 
y = np.delete(test_set_NM, 108, axis=1)
regn = Linear_Regression_nobias()
reg = Linear_Regression()
#%%
weight_b = regn.get_weight(x, y_train)
y_b = regn.predict(y, weight_b)
RMSE_b = rmse(y_b, y_test)
print("RMSE_b =" , RMSE_b)
#%%
# =============================================================================
# plot
# =============================================================================

