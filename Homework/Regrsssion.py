# -*- coding: utf-8 -*-
"""
20201017

Editor:Hsieh

"""
#%%
import pandas as pd
import numpy as np
import sympy as sp
import matplotlib.pyplot as plt

#%%
# =============================================================================
# split_data
# =============================================================================

data = pd.read_csv(r"C:\Users\user\Desktop\NTU\Fintech\HW1\train.csv")
df = pd.get_dummies(data)  #one-hot encoding
data_set = np.array(df)
index = np.array(data["ID"]-1)
# np.random.shuffle(index)
training_set = np.empty([int(data_set.shape[0]*0.8), data_set.shape[1]])
test_set = np.empty([int(data_set.shape[0]*0.2), data_set.shape[1]])
m=0 ;n= 0
for idx in index[:training_set.shape[0]]:
    training_set[m]=data_set[idx]
    m +=1
for idx in index[training_set.shape[0]:]:
    test_set[n]=data_set[idx]
    n +=1
mean = training_set.mean(axis=0) ;std = training_set.std(axis=0)
training_set_NM = (training_set-mean)/std
test_set_NM = (test_set-mean)/std

G3_train_NM = training_set_NM[:,16]
G3_test_NM =  test_set_NM[:,16]
G3_train = training_set[:,16]
G3_test =  test_set[:,16]
    
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
  
class Linear_Regression_nobias:   #no bias
 
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
x = np.delete(training_set_NM, 16, axis=1) #remove G3 from training_set
y = np.delete(test_set_NM, 16, axis=1) #remove G3 from test_set 
regn = Linear_Regression_nobias()
reg = Linear_Regression()
# =============================================================================
# 1.(b)
# =============================================================================
#%%
weight_b = regn.get_weight(x, G3_train)
G3_b = regn.predict(y, weight_b)
RMSE_b = rmse(G3_b, G3_test)
print("RMSE_b =" , RMSE_b)
#%%
# =============================================================================
# 1.(c)
# =============================================================================
weight = regn.get_weight(x, G3_train)
mse_train = mse(regn.predict(x, weight), G3_train)
w = sp.symbols("w:62")
wm = sp.Matrix(w)
Jm = (1/2)*np.matmul(wm.T, wm) + sp.Matrix([mse_train])
J_p = sp.diff(w , Jm)
# weight_c = 

G3_c = regn.predict(y, weight_c)
RMSE_c = rmse(G3_c, G3_test)
print("RMSE_c =" , RMSE_c)
#%%
# =============================================================================
# 1.(d)
# =============================================================================
weight = reg.get_weight(x, G3_train)
a= mse(reg.predict(x, weight), G3_train)+ (1/2)*(weight.T*weight)
G3_d = reg.predict(y, a)
RMSE_d = rmse(G3_d, G3_test)
print("RMSE_d =" , RMSE_d)
#%%
# =============================================================================
# 1.(e)
# =============================================================================
weight = reg.get_weight(x, G3_train)
A0 = np.identity(62)
x_shape = x.shape
X= np.concatenate([np.ones((x_shape[0],1)), x], axis=1)
A = np.linalg.inv(np.matmul(X.T,X) + np.linalg.inv(A0)*(1/10))
u = np.matmul(A, (np.matmul(X.T, G3_train )+ 0*np.linalg.inv(A0)))
weight_e = (-1/2)* np.matmul(np.matmul((weight-u).T, np.linalg.inv(A)), (weight-u))
G3_e = reg.predict(y, weight_e)
RMSE_e = rmse(G3_e, G3_test)
print("RMSE_e =" , RMSE_e)





#%%
# =============================================================================
# plot
# =============================================================================
plt.plot(index[0:200],G3_test, label="ground truth")
plt.plot(index[0:200],G3_b, label="Linear Regression")
# plt.plot(index[0:200],G3_c)
# plt.plot(index[0:200],G3_d, label="3")
# plt.ylim(-15,20)
plt.legend()
plt.show()



