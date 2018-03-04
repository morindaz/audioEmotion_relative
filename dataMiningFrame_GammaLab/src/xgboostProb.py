# coding=utf-8
'''
使用xgboost计算top5的accuracy
使用softprob
'''
from loadData import *
import xgboost as xgb
import pandas as pd
import numpy as np
from sklearn.externals import joblib
import os
#数组的顺序按照0-5排序
params={
'booster':'gbtree',
# 这里手写数字是0-9，是一个多类的问题，因此采用了multisoft多分类器，
'objective': 'multi:softprob',
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
    myfile = os.path.exists(modelPath+"xg_model_prob.model")
    if myfile:
        print('==== 参数最优化已经存在.')
    else:
        model_prob = xgb.train(plst, xg_train, num_rounds, watchlist,early_stopping_rounds=100)
        joblib.dump(model_prob, modelPath + "xg_model_prob.model")
    clf = joblib.load(modelPath + "xg_model_prob.model")
    preds_prob = clf.predict(xg_test,ntree_limit=clf.best_iteration)
    return preds_prob

preds_prob = formalTrain()
k = 6
result = []
for i in preds_prob:
    b = i.argsort()[-k:][::-1]
    print (b)
    result.append(b)
preds_prob =np.array(preds_prob)
result = np.array(result)
print (result)
def output_prob():
    myfileCSV = os.path.exists('preds_prob_sort.csv')
    if myfileCSV:
        print('==== CSV数据已经存在.')
    else:
        #经过排序后的概率emotionOutput
        emotionOutput = {"top1":result[:,0],"top2":result[:,1],"top3":result[:,2],"top4":result[:,3],"top5":result[:,4]}
        #没有经过排序的概率outPut
        outPut = {'prob0': preds_prob[:,0],'prob1': preds_prob[:,1],'prob2': preds_prob[:,2],'prob3': preds_prob[:,3],'prob4': preds_prob[:,4],'prob5': preds_prob[:,5]}
        output_emotion = pd.DataFrame(emotionOutput)
        output_Archive = pd.DataFrame(outPut)
        output_emotion.to_csv('emotion.csv')
        output_Archive.to_csv('preds_prob_sort.csv')

#计算prob的正确率
def accuracyCount(param, name):
    count = 0
    allCnt = 0
    for i in range(len(y_test)):
        if y_test[i] in param[i]:
            count = count + 1
        allCnt = allCnt + 1
    accuracy = float(count) / float(len(y_test))
    print('=========This is:%s' % name)
    print(accuracy)
result = np.array(result)
tops = []
for i in range(2,7):
    namestr = "top"+str(i)
    name = result[:,0:i]
    accuracyCount(name,namestr)