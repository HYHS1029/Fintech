# -*- coding: utf-8 -*-
"""
Created on Tue Dec 22 17:09:20 2020

@author: Hsin-Yuan
"""
import pandas as pd
import numpy as np
import mplfinance as mpf
import talib
import matplotlib.pyplot as plt
from keras.layers import Dense, Dropout
from keras.layers import SimpleRNN
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

#%% plot Trading chart 
K = mpf.make_addplot(df['k'], panel = 2)
D = mpf.make_addplot(df['d'], panel = 2)
ma10 = mpf.make_addplot(df['ma_10'])
ma30 = mpf.make_addplot(df['ma_30'])

mpf.plot(
         df, 
         type = 'candle', 
         style = 'yahoo',
         ylabel = 'Price',
         title = '2019-01-01 ~ 2019-12-31 Trading chart',
         addplot = [K,D,ma10,ma30],
         volume = True)

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
#%% simple RNN
def buildsimpleRNN(shape):
  model = Sequential()
  model.add(SimpleRNN(9, input_shape = (shape[1], shape[2])))
  model.add(Dropout(0.3))
  model.add(Dense(1))   
  model.compile(loss="mse", optimizer="adam")
  model.summary()
  return model

RNN = buildsimpleRNN(X_train.shape)
#%% simpleRNN result
RNN_history = RNN.fit(X_train, Y_train, validation_data=(X_val, Y_val), batch_size=16, epochs=20)
y_hat = RNN.predict(X_val)
plt.figure()
plt.plot(RNN_history.history['loss'])
plt.plot(RNN_history.history['val_loss'])
plt.title('simpleRNN loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['train', 'val'], loc='upper right')
plt.figure()
plt.plot(Y_val)
plt.plot(y_hat, 'r--')
plt.title('simpleRNN prediction')
plt.legend(['real', 'predict'], loc='upper right')

