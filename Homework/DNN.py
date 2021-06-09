#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: hsin-yuan
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sn
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
from keras import optimizers
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score, auc
from sklearn.metrics import precision_recall_curve, f1_score
import scikitplot as skplt
#%%
df = pd.read_csv(r"C:\Users\user\Desktop\NTU\Fintech\HW2\Data.csv")
x = df.iloc[:, 0:31]
y = df.iloc[:, -1] 
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
#%%
model = Sequential()
#Hidden Layer
model.add(Dense(64, input_dim=31, activation='relu', kernel_initializer='uniform'))
#Second  Hidden Layer
model.add(Dense(32, activation='relu', kernel_initializer='uniform'))
#Output Layer
model.add(Dense(1, activation='sigmoid', kernel_initializer='uniform'))
#Compiling the neural network
Adam = optimizers.Adam(lr= 0.001)
model.compile(optimizer = Adam, loss='binary_crossentropy', metrics =['accuracy'])
#%%
history = model.fit(X_train, y_train, validation_data=(X_test, y_test), batch_size=10, epochs=50)
print(history.history.keys())
#%%
plt.figure()
plt.plot(history.history['accuracy'], label ='train')
plt.plot(history.history['val_accuracy'], label ='validation')
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend()
plt.show()

plt.figure()
plt.plot(history.history['loss'], label ='train')
plt.plot(history.history['val_loss'], label ='validation')
plt.title('model loss')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend()
plt.show()
#%%  
y_test_pred = model.predict(X_test)
y_test_pred = (y_test_pred>0.5)
y_train_pred = model.predict(X_train)
y_train_pred = (y_train_pred>0.5)
#%%
cm_t = confusion_matrix(y_train, y_train_pred)
cm_v = confusion_matrix(y_test ,y_test_pred)
sn.set(font_scale=1.4) 
plt.figure()
sn.heatmap(cm_v, annot=True, fmt = 'g') 
plt.title('validation confusion matrix')
plt.figure()
sn.heatmap(cm_t, annot=True, fmt = 'g') 
plt.title('train confusion matrix')
#%%
precision = cm_v[0,0]/(cm_v[0,0]+cm_v[0,1])
recall = cm_v[0,0]/(cm_v[0,0]+cm_v[1,0])
F1_score = 2*((precision*recall)/(precision+recall))
print(' precision = {} \n recall = {} \n F1_score = {}'.format(precision, recall, F1_score))
#%%
# plot the roc curve for the model
ns_probs = [0 for _ in range(len(y_test))]
model_probs = model.predict_proba(X_test)
model_probs = model_probs.flatten() 
ns_auc = roc_auc_score(y_test, ns_probs)
model_auc = roc_auc_score(y_test, model_probs)
print('No Skill: ROC AUC=%.3f' % (ns_auc))
print('DNN: ROC AUC=%.3f' % (model_auc))
ns_fpr, ns_tpr, _ = roc_curve(y_test, ns_probs)
model_fpr, model_tpr, _ = roc_curve(y_test, model_probs)
# plot the roc curve for the model
plt.plot(ns_fpr, ns_tpr, linestyle='--')
plt.plot(model_fpr, model_tpr, marker='.', label='AUC={}'.format(model_auc))
plt.title('ROC curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
plt.show()
#%%
## plot the precision-recall curves
model_precision, model_recall, _ = precision_recall_curve(y_test, model_probs)
model_f1, model_auc = f1_score(y_test, y_test_pred), auc(model_recall, model_precision)
print('DNN: f1=%.3f auc=%.3f' % (model_f1, model_auc))
no_skill = len(y_test[y_test==1]) / len(y_test)
plt.plot([0, 1], [no_skill, no_skill], linestyle='--')
plt.plot(model_recall, model_precision, marker='.')
plt.title('PRC')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.legend()
plt.show()
#%%
#plot Lift curve











