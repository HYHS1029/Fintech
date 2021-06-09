# -*- coding: utf-8 -*-
"""
Created on Tue Dec 22 17:09:20 2020

@author: Hsin-Yuan
"""
import pandas as pd
import numpy as np
import talib
import matplotlib.pyplot as plt
from keras.layers import Dense, Dropout
from keras.layers import GRU
from keras.models import Sequential
#%%
def normalize(x):
    return (x - x.min() ) / (x.max()- x.min())
#%% read data
data = pd.read_csv(r"C:\Users\User\OneDrive\桌面\Fintech\HW4\S_P.csv", index_col=0,parse_dates=True)

# add feature (MA 10 days, MA 30 days, K, D) and normalize the data
data['ma_10'] = talib.MA(data['Close'], timeperiod = 10, matype=0)
data['ma_30'] = talib.MA(data['Close'], timeperiod = 30, matype=0)
data['k'], data['d'] = talib.STOCH(data['High'], data['Low'], data['Close'])
data_nor = normalize(data)

# define the training data and validation data
plot_range = pd.date_range(start="2019-01-01", end="2019-12-31")
train_range = pd.date_range(start="1994-01-01", end="2017-12-31")
val_range = pd.date_range(start="2018-01-01", end="2019-12-31")

df = data[data.index.isin(plot_range)]
train_set = data_nor[data_nor.index.isin(train_range)]
val_set = data_nor[data_nor.index.isin(val_range)]
#%% Build Data
def build_data(data, timestep, futureDay):
  X, Y = [], []
  data_noclose = data.drop(['Close'], axis= 1)
  for i in range(data.shape[0]- futureDay- timestep):
    X.append(np.array(data_noclose.iloc[i:i+ timestep]))
    Y.append(np.array(data.iloc[i+timestep : i+timestep +futureDay]["Close"]))
    
  return np.array(X), np.array(Y)

train_set = train_set.fillna(0)
X_train, Y_train = build_data(train_set, 30, 1)
X_val, Y_val = build_data(val_set, 30, 1)

# GRU model
def buildGRU(shape):
  model = Sequential()
  model.add(GRU(9, input_shape = (shape[1], shape[2])))
  model.add(Dropout(0.3))
  model.add(Dense(1))
  model.compile(loss="mse", optimizer="adam")
  model.summary()
  return model

gru = buildGRU(X_train.shape)
#%% GRU result
GRU_history = gru.fit(X_train, Y_train, validation_data=(X_val, Y_val), batch_size=16, epochs=20)
y_hat = gru.predict(X_val)
plt.figure()
plt.plot(GRU_history.history['loss'])
plt.plot(GRU_history.history['val_loss'])
plt.title('GRU loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['train', 'val'], loc='upper right')
plt.figure()
plt.plot(Y_val)
plt.plot(y_hat, 'r--')
plt.title('GRU prediction')
plt.legend(['real', 'predict'], loc='upper right')
