#!/usr/bin/env python
# coding: utf-8



import pandas as pd
# import mplfinance as mpf
import datetime
from datetime import date
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import tensorflow as tf
import keras
from keras.layers import LSTM
from keras.layers import Dense, Activation, Conv2D, MaxPool2D, Dropout, Flatten, SimpleRNN, GRU
from keras.datasets import mnist
from keras.models import Sequential
from keras.optimizers import Adam
from keras.layers import Dropout
import matplotlib.pyplot as plt
# import lightgbm as lgb

def transform_date(date):   #民國轉西元
    y, m, d = date.split('/')
    return str(int(y)+1911) + '/' + m  + '/' + d

def transform(data):   #讀取每一個元素進行資料格式轉換，再產生新的串列
    return [transform_date(d) for d in data]


def create_GRU(x_train, y_train, x_test, y_test):
    batch_size = None
    steps = 30
    input_dim = x_train.shape[2]
    epochs = 60

    model = Sequential()
  # 加 RNN 隱藏層(hidden layer)
    model.add(GRU(
      
    batch_input_shape=(batch_size, steps, input_dim), 
    units= 50,
    unroll=True,
    return_sequences=True
  ))
    model.add(GRU(
      units= 50,
      unroll=True,
  ))
    model.add(Dropout(0.2))
    model.add(Dense(20))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mean_squared_error'])
    hist = model.fit(x_train, y_train,
            batch_size=batch_size, epochs=epochs,
            verbose=1, validation_data=(x_test, y_test))

    return hist, model

def plot_loss(hist, modelname):
    name = modelname+" loss"
    plt.plot(hist.history['loss'])
    plt.plot(hist.history['val_loss'])
    plt.legend(['Train', 'Test'], loc='upper right')
    plt.title(modelname+" loss")
    plt.savefig(name)
    plt.show()

def denormalized(data, data_normalize):
    denormal = data_normalize*(data['close'].max()-data['close'].min())+data['close'].min()
    return denormal

def plot_predict_close(x_test, y_test, model, modelname, data):
    predict_y = model.predict(x_test)
  
    denormal_predict_y = denormalized(data, predict_y)
    denormal_y = denormalized(data, y_test)
    name = modelname+' prediction'
    plt.plot(denormal_y)
    plt.plot(denormal_predict_y)
    plt.legend(['True', 'Predict'], loc='upper right')
    plt.title(name)
    plt.savefig(name)
    plt.show()




data = pd.read_csv('final.csv',parse_dates=True)
data['date']= pd.to_datetime(data['date'])

notnumber_idx = np.where(data['price_difference'] == 'X0.00')
for idx in notnumber_idx:
  # print(idx)
  data.iloc[idx] = 0

data.set_index('date',inplace=True)
data['price_difference'] = data['price_difference'].astype(float)




_ = data.pop('snownlp')
_ = data.pop('bert')
data_normalize = (data-data.min())/(data.max()-data.min())

# idx = (data_normalize.shape[0]*3)//4
idx = 300
trainData = data_normalize.iloc[:idx,:]
testData = data_normalize.iloc[idx:, :]
y_trainData = trainData.pop('close')
y_testData = testData.pop('close')

data_normalize




x_train = []   #預測點的前 days 天的資料
y_train = []   #預測點
x_test = []
y_test = []
days = 30
for i in range(days, trainData.shape[0]):  # trainData.shape[0] 是訓練集總數
    x_train.append(trainData.iloc[i-days:i, :])
    y_train.append(y_trainData.iloc[i])
for i in range(days, testData.shape[0]):
    x_test.append(testData.iloc[i-days:i, :])
    y_test.append(y_testData.iloc[i])
x_train, y_train = np.array(x_train), np.array(y_train)  # 轉成numpy array的格式，以利輸入 RNN
x_test, y_test = np.array(x_test), np.array(y_test)
# y_testData 




GRU_hist, GRU = create_GRU(x_train, y_train, x_test, y_test)




plot_loss(GRU_hist, 'GRU')




plot_predict_close(x_test, y_test, GRU, "GRU", data)




def var_importance(model,x):
    orig_out = model.predict(x)
    for i in range(x_train.shape[2]):  # iterate over the three features
        new_x = x.copy()
        perturbation = np.random.normal(0.0, 0.2, size=new_x.shape[:2])
        new_x[:, :, i] = new_x[:, :, i] + perturbation
        perturbed_out = model.predict(new_x)
        effect = ((orig_out - perturbed_out) ** 2).mean() ** 0.5
    
    print(trainData.columns[i]+ f' perturbation effect: {effect:.4f}')




var_importance(GRU, x_train)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
import datetime as datetime


final = pd.read_csv('final.csv')




final.head()



sent = pd.DataFrame(final,columns = ['date','price_difference','snownlp','bert','close'])



sent['price_difference'] = sent['price_difference'].astype('float')
sent['close'] = sent['close'].astype('float')



sent['Quote change'] = (sent['price_difference']/(sent['close']-sent['price_difference']))*100



sent = sent.drop(columns = ['close'])



train = sent.loc[sent['date'].str[:4].astype(int)<=2019]
test = sent.loc[sent['date'].str[:4].astype(int)>2019]



train.index = train['date']
train = train.drop(columns = ['date'])



test.index = test['date']
test = test.drop(columns = ['date'])




train = (train - train.min()) / (train.max() - train.min())




test = (test - test.min()) / (test.max() - test.min())




from keras.models import Sequential
from keras.layers import Dense, Activation




import numpy as np




def buildTrain(train, pastDay=30, futureDay=1):
    X_train, y_train = [], []
    for i in range(train.shape[0]-futureDay-pastDay):
        X_train.append(np.array(train.iloc[i:i+pastDay]))
        y_train.append(np.array(train.iloc[i+pastDay:i+pastDay+futureDay]["Quote change"]))
    return np.array(X_train), np.array(y_train)




X_train, y_train = buildTrain(train, 30, 1)




X_val, y_val = buildTrain(test, 30, 1)




from keras.layers import SimpleRNN, LSTM, GRU




model = Sequential()
model.add(LSTM(1, input_shape = (30,4)))

model.compile(optimizer='adam', loss='mean_squared_error')
model.summary()




train_history = model.fit(X_train, y_train, epochs=20, batch_size=128, validation_data=(X_val, y_val))




def show_Learning_Curve(train_loss,test_loss):
    plt.plot(train_history.history[train_loss])
    plt.plot(train_history.history[test_loss])
    plt.title('Learning Curve')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['train loss', 'test loss'], loc='upper left')
    plt.show()
show_Learning_Curve('loss','val_loss')




pred = model.predict(X_val)
pred = pd.DataFrame(pred,columns=["pred"])




com = test['Quote change'] 
com = com.reset_index()
com = com.drop(com.index[0:31])
com.index = com['date']
com = com.drop(columns=['date'])




compare = pd.concat([com, pred.set_index(com.index)], axis=1)




def show_prediction(Quote_change , pred):
    fig, ax = plt.subplots()
    ax.set_xticks(range(0, len(compare.index), 30))
    ax.set_xticklabels(compare.index[::30])
    plt.plot(Quote_change)
    plt.plot(pred)
    plt.title('prediction of Quote change ')
    plt.xlabel('date')
    plt.legend(['Quote change', 'pred'], loc='upper left')
    plt.show()
show_prediction(compare['Quote change'],compare['pred'])





train1 = sent.loc[sent['date'].str[:4].astype(int)<=2019]
test1 = sent.loc[sent['date'].str[:4].astype(int)>2019]





train1.index = train1['date']
train1 = train1.drop(columns = ['date','bert'])





test1.index = test1['date']
test1 = test1.drop(columns = ['date','bert'])





train1 = (train1 - train1.min()) / (train1.max() - train1.min())





test1 = (test1 - test1.min()) / (test1.max() - test1.min())





X_train1, y_train1 = buildTrain(train1, 30, 1)





X_val1, y_val1 = buildTrain(test1, 30, 1)





model = Sequential()
model.add(GRU(1, input_shape = (30,3)))

model.compile(optimizer='adam', loss='mean_squared_error')
model.summary()

train_history = model.fit(X_train1, y_train1, epochs=20, batch_size=128, validation_data=(X_val1, y_val1))

pred1 = model.predict(X_val1)
pred1 = pd.DataFrame(pred1,columns=["pred1"])
com1 = test1['Quote change'] 
com1 = com1.reset_index()
com1 = com1.drop(com1.index[0:31])
com1.index = com1['date']
com1 = com1.drop(columns=['date'])
compare1 = pd.concat([com1, pred1.set_index(com1.index)], axis=1)
show_prediction(compare1['Quote change'],compare1['pred1'])

train2 = sent.loc[sent['date'].str[:4].astype(int)<=2019]
test2 = sent.loc[sent['date'].str[:4].astype(int)>2019]
train2.index = train2['date']
train2 = train2.drop(columns = ['date','snownlp'])
test2.index = test2['date']
test2 = test2.drop(columns = ['date','snownlp'])
train2 = (train2 - train2.min()) / (train2.max() - train2.min())
test2 = (test2 - test2.min()) / (test2.max() - test2.min())

X_train2, y_train2 = buildTrain(train2, 30, 1)

X_val2, y_val2 = buildTrain(test2, 30, 1)

model = Sequential()
model.add(GRU(1, input_shape = (30,3)))

model.compile(optimizer='adam', loss='mean_squared_error')
model.summary()

train_history = model.fit(X_train2, y_train2, epochs=20, batch_size=128, validation_data=(X_val2, y_val2))

pred2 = model.predict(X_val2)
pred2 = pd.DataFrame(pred2,columns=["pred2"])

com2 = test2['Quote change'] 
com2 = com2.reset_index()
com2 = com2.drop(com2.index[0:31])
com2.index = com2['date']
com2 = com2.drop(columns=['date'])

compare2 = pd.concat([com2, pred2.set_index(com2.index)], axis=1)

show_prediction(compare2['Quote change'],compare2['pred2'])



