# coding=utf-8
'''
tune models
此文件用于选择参数，一个模型可能有10个参数，需要先固定哪几个参数

'''
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import recall_score, precision_score, make_scorer, roc_auc_score, confusion_matrix
from sklearn.svm import LinearSVC
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier

from sklearn.naive_bayes import MultinomialNB
from sklearn import pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from loadData import *

def scoring_method(y_true, y_pred, pos_label=1):
    return recall_score(y_true, y_pred, pos_label=pos_label,average='weighted')

# Params List
# modelName = 'linearsvc'
methodList= ["SVM","LR","RF","xgBoost"]
methodChoice = methodList[3] #选择何种方法

def model_params(model):
    param_gridA = {}
    method = ""
    if model =="SVM":
        cRange = np.logspace(0, 2, 6)
        penaltyRange = ['l1', 'l2']
        maxIterRange = np.logspace(2, 4, 5)
        param_gridA = {'linearsvc__C': cRange,
                       'linearsvc__penalty': penaltyRange,
                       'linearsvc__max_iter': maxIterRange}
        method=LinearSVC(dual=False, class_weight='balanced')
    elif model=="LR":
        cRange = np.logspace(0, 2, 6)
        param_gridA = {'logisticregression__C': cRange}
        method=LogisticRegression(penalty='l2', multi_class='multinomial', class_weight='balanced', solver='lbfgs')
    elif model=="xgBoost":
        param_gridA = {
                        'xgbclassifier__num_class':[0, 10, 10]}
        method = xgb.XGBClassifier(booster='gbtree',num_class = 20,n_jobs = 1,eta = 0.05,num_boost_round = 2,learning_rate=0.1,
        max_depth = 6,subsample = 0.9,colsample_bytree = 0.9,silent = 0,objective='multi:softmax',cv=5,gamma=0.07,n_estimator=55,min_child_weight=3,seed=710)
    print method
    return param_gridA,method


# 找到模型最优参数
def searchMethodFun():
    weightRange = [dict(zip(range(0, 2), (1, values))) for values in range(14, 24)]
    param_gridA, method = model_params(methodChoice)
    pipeA = pipeline.make_pipeline(preprocessing.StandardScaler(), method)
    # 产生指定数量的独立的train/test数据集划分, 首先对样本全体打乱, 然后划分出train／test对
    # 返回分层划分， 创建划分的时候要保证每一个划分中类的样本比例与整体数据集中的原始比例保持一致
    fscore = make_scorer(scoring_method, pos_label=1)
    sss = StratifiedShuffleSplit(n_splits=3, test_size=0.5, random_state=0)
    sss.get_n_splits(y_train)
    searchMethod = GridSearchCV(pipeA, param_grid=param_gridA, scoring=fscore, cv=sss, n_jobs=1)
    # searchMethod = RandomizedSearchCV(pipeA, param_distributions=param_gridA,n_iter=20)
    return searchMethod
