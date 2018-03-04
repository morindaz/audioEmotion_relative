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

df = pd.read_csv(csvPath, sep=',', header=0)
# data = df.values[:, 0:-1]
data = df.values
train,test = train_test_split(data,test_size=0.25, random_state=0)
trainData = pd.DataFrame(train)
testData = pd.DataFrame(test)
trainData.to_csv("train.csv")
testData.to_csv("test.csv")
tempdf = pd.read_csv(featurePath, sep=',', header=0)
tempFeature = tempdf.values[:, -1]
featureList = []
for i in tempFeature:
    featureList.append(i)
<<<<<<< HEAD
# print featureList
=======
print (featureList)
>>>>>>> 53305f2d5a9c69776fc4886f92d85535ea8d3da8
X = data[:, featureList]
X = preprocessing.scale(X)
Y = np.array(map(int, df.values[:, -1]))
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.25, random_state=0)
# X_train, X_train_lr, y_train, y_train_lr = train_test_split(X_train,y_train,test_size=0.5)
mms = preprocessing.StandardScaler()
X_train = mms.fit_transform(X_train)
# X_train_lr = mms.fit_transform(X_train_lr)
X_test = mms.transform(X_test)
# trainCombine = np.column_stack((X_train,y_train))
# testCombine = np.column_stack((X_test,y_test))
trainData = pd.DataFrame(train)
testData = pd.DataFrame(test)
trainData.to_csv("train.csv")
testData.to_csv("test.csv")

