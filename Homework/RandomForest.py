# -*- coding: utf-8 -*-
"""

@author: Hsin-yuan
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score, auc
from sklearn.metrics import precision_recall_curve, f1_score
#%%
df = pd.read_csv(r"C:\Users\user\Desktop\NTU\Fintech\HW2\Data.csv")
x = df.iloc[:, 0:31]
y = df.iloc[:, -1] 
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
#%%
# =============================================================================
# random forest
# =============================================================================
forest = RandomForestClassifier(criterion = 'entropy', n_estimators=10, random_state=3, n_jobs=2)
history = forest.fit(X_train, y_train)
y_test_pred = forest.predict(X_test)
y_train_pred = forest.predict(X_train)
#%%
cm = confusion_matrix(y_test ,y_test_pred)
precision = cm[0,0]/(cm[0,0]+cm[0,1])
recall = cm[0,0]/(cm[0,0]+cm[1,0])
F1_score = 2*((precision*recall)/(precision+recall))
print(' precision = {} \n recall = {} \n F1_score = {}'.format(precision, recall, F1_score))
sn.set(font_scale=1.4) 
plt.figure()
sn.heatmap(cm, annot=True, fmt = 'g') 
plt.title('confusion matrix')
#%%
# plot the roc curve for the model
ns_probs = [0 for _ in range(len(y_test))]
model_probs = forest.predict_proba(X_test)
model_probs = model_probs[:, 1]
ns_auc = roc_auc_score(y_test, ns_probs)
model_auc = roc_auc_score(y_test, model_probs)
print('No Skill: ROC AUC=%.3f' % (ns_auc))
print('Random forest: ROC AUC=%.3f' % (model_auc))
ns_fpr, ns_tpr, _ = roc_curve(y_test, ns_probs)
model_fpr, model_tpr, _ = roc_curve(y_test, model_probs)
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
print('Random forest: f1=%.3f auc=%.3f' % (model_f1, model_auc))
no_skill = len(y_test[y_test==1]) / len(y_test)
plt.plot([0, 1], [no_skill, no_skill], linestyle='--')
plt.plot(model_recall, model_precision, marker='.')
plt.title('PRC')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.legend()
plt.show()
