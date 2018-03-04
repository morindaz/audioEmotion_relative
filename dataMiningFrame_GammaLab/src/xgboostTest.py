# coding=utf-8
'''
使用xgboost predict类别
使用softmax
'''
from loadData import *
import xgboost as xgb
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.externals import joblib
import os

#数组的顺序按照0-5排序
params={
'booster':'gbtree',
# 这里手写数字是0-9，是一个多类的问题，因此采用了multisoft多分类器，
'objective': 'multi:softmax',
'num_class':60, # 类数，与 multisoftmax 并用
'gamma':0.07,  # 在树的叶子节点下一个分区的最小损失，越大算法模型越保守 。[0:]
'max_depth':6, # 构建树的深度 [1:]
#'lambda':450,  # L2 正则项权重
'subsample':0.9, # 采样训练数据，设置为0.5，随机选择一般的数据实例 (0:1]
'colsample_bytree':0.9, # 构建树树时的采样比率 (0:1]
#'min_child_weight':12, # 节点的最少特征数
'verbose':1,
'silent':0 ,
'eta': 0.05, # 如同学习率
'seed':710,
'nthread':1,# cpu 线程数,根据自己U的个数适当调整
'n_estimator':55,
'learning_rate':0.01,
'min_child_weight':3,
'scale_pos_weight':1
}

num_rounds = 2 # 迭代次数
def formalTrain():
    modelPath = "../models/"
    xg_train = xgb.DMatrix(X_train, label=y_train)
    xg_test = xgb.DMatrix(X_test, label=y_test)
    # setup parameters for xgboost
    plst = list(params.items())
    watchlist = [(xg_train, 'train'),(xg_test, 'test')]
    myfile = os.path.exists(modelPath+"xg_model.model")
    if myfile:
        print('==== 参数最优化已经存在.')
    else:
        model_prob = xgb.train(plst, xg_train, num_rounds, watchlist,early_stopping_rounds=100)
        joblib.dump(model_prob, modelPath + "xg_model.model")
    clf = joblib.load(modelPath + "xg_model.model")
    preds_prob = clf.predict(xg_test,ntree_limit=clf.best_iteration)
    return preds_prob
preds_prob = formalTrain()

acc = accuracy_score(y_test, preds_prob)
macroF1 = f1_score(y_test, preds_prob, average='macro')
microF1 = f1_score(y_test, preds_prob, average='micro')
weightedF1 = f1_score(y_test, preds_prob, average='weighted')
eachClass = f1_score(y_test, preds_prob, average=None)
print('==== The Accuracy is:%s' % acc)
print('==== The macroF1 is:%s' % macroF1)
print('==== The microF1 is:%s' % microF1)
print('==== The weightedF1 is:%s' % weightedF1)
print('==== The eachClassF1 is:%s' % eachClass)

