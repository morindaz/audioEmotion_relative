# coding=utf-8
'''
loadData.py
文件加载dependence.py文件，获取相关数据路径、特征等参数
此文件的数据，在训练和预测时候都需要用
featureSelection和modelTrain都需要首先加载他
paraRange里面负责参数调整
'''
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.cross_validation import train_test_split
from dependence import *
import os
import numpy as np
mingw_path = 'C:\\Program Files\\mingw-w64\\x86_64-5.3.0-posix-seh-rt_v4-rev0\\mingw64\\bin'
os.environ['PATH'] = mingw_path + ';' + os.environ['PATH']

test = pd.read_csv(testPath, sep=',', header=0)
train = pd.read_csv(trainPath, sep=',', header=0)
dataTest = test.values[:, 0:-1]
dataTrain = train.values[:, 0:-1]
tempdf = pd.read_csv(featurePath, sep=',', header=0)
tempFeature = tempdf.values[:, -1]
featureList = []
for i in tempFeature:
    featureList.append(i)

print (featureList)
X_test = dataTest[:, featureList]
# X_test = dataTest[:,0:978]
X_train = dataTrain[:, featureList]
# X_train = dataTrain[:,0:978]
X_test2 = preprocessing.scale(X_test)
X_train2 = preprocessing.scale(X_train)
# y_test = np.array(map(int, test.values[:, -1]))
# y_train = np.array(map(int, train.values[:, -1]))
y_test = test.values[:, -1]
y_train = train.values[:, -1]
mms = preprocessing.StandardScaler()
X_train = mms.fit_transform(X_train)
# X_train_lr = mms.fit_transform(X_train_lr)
X_test = mms.transform(X_test)

# print (y_train)