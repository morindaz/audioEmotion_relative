# coding=utf-8
import numpy as np
np.random.seed(10)

import matplotlib.pyplot as plt
from sklearn.metrics import f1_score
from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (RandomTreesEmbedding, RandomForestClassifier,
                              GradientBoostingClassifier)
from hyperopt import fmin,tpe,hp,partial
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve
from sklearn.pipeline import make_pipeline
from loadData import *
from sklearn.svm import LinearSVC

from sklearn.grid_search import GridSearchCV

# best hyperparameter setting
def GridSearchGBDT():
    param_grid = {'learning_rate': [0.1, 0.05, 0.02, 0.01],
                  'max_depth': [4, 6],
                  'min_samples_leaf': [3, 5, 9, 17],
                  # 'max_features': [1.0, 0.3, 0.1] ## not         possible in our example (only 1 fx)
                  }
    #GridSearch method
    est = GradientBoostingClassifier(n_estimators=10)
    X_train_scaled = mms.fit_transform(X_train)
    gs_cv = GridSearchCV(est, param_grid, n_jobs=1).fit(X_train_scaled, y_train)
    best_params =  gs_cv.best_params_
    best_estimator = gs_cv.best_estimator_
    return best_estimator,best_params

n_estimator = 10
m,n = GridSearchGBDT()
print m
print n
# Unsupervised transformation based on totally random trees
rt = RandomTreesEmbedding(max_depth=3, n_estimators=10,random_state=0)
rt_lm = LogisticRegression()

# Supervised transformation based on random forests

#可用方法1
# grd = GradientBoostingClassifier(n_estimators=n_estimator,learning_rate=0.02, max_depth=4, min_samples_leaf=17)
# grd_enc = OneHotEncoder()
# grd_lm = LogisticRegression()
# grd.fit(X_train, y_train)
# grd_enc.fit(grd.apply(X_train)[:, :, 0])
# grd_lm.fit(grd_enc.transform(grd.apply(X_train_lr)[:, :, 0]), y_train_lr)
# y_pred_grd_lm = grd_lm.predict_proba(grd_enc.transform(grd.apply(X_test)[:, :, 0]))[:, 1]

#可用方法2
# rf = GradientBoostingClassifier(n_estimators=n_estimator,learning_rate=0.02, max_depth=4, min_samples_leaf=17)
# rf_enc = OneHotEncoder()
# # rf_lm = LogisticRegression()
# rf_lm = LinearSVC()
# rf.fit(X_train, y_train)
# rf_enc.fit(rf.apply(X_train))
# rf_lm.fit(rf_enc.transform(rf.apply(X_train_lr)), y_train_lr)
# y_pred_rf_lm = rf_lm.predict_proba(rf_enc.transform(rf.apply(X_test)))[:, 1]
# y_pred = grd_lm.predict(grd_enc.transform(grd.apply(X_test)[:, :, 0]))
# acc = accuracy_score(y_test, y_pred)
# print y_pred
# print('==== The Accuracy is:%s' % acc)

# # clf = GradientBoostingClassifier(n_estimators=n_estimator,learning_rate=0.05, max_depth=4, min_samples_leaf=17)
clf = GradientBoostingClassifier(n_estimators=n_estimator,learning_rate=0.02, max_depth=4, min_samples_leaf=17)
# clf = GradientBoostingClassifier(n_estimators=n_estimator,learning_rate=0.02, max_depth=4, min_samples_leaf=17)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
# # this may take some minutes

print y_pred
acc = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average='weighted')
f11 = metrics.precision_score(y_test, y_pred, labels=[0, 1, 2, 3, 4, 5], average='macro')
print('==== The Accuracy is:%s' % acc)
print('==== The F1_score is:%s' % f1)
print('==== The F11_score is:%s' % f11)