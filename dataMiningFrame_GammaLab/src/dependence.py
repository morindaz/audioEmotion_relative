# coding=utf-8
'''
此文件中记录一些常用的配置文件，包含：
读入数据的位置csvPath
特征存放位置featurePath
模型位置modelPath+estimatorName
模型参数位置modelPath+paraName
'''

csvPath ='../data/selectedCombinedAllwithLabel.csv'  #读入数据的位置
testPath = '../data/test.csv'
trainPath = '../data/train.csv'
featureBasic = '..//feature//'
featureName = 'cv0Clear30.csv' #特征的具体csv名字
featurePath = featureBasic+featureName
modelPath = '..//models//'
estimatorName = "estimator.model"
paramName = "param.pkl"
