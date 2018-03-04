# encoding:UTF-8
# import h5py
'''
ModelTrain 负责训练SVM分类器
通过gridSearch寻找模型最优参数
'./param.pkl'  './estimator.model' 保存了最优参数以及模型
'''

import matplotlib.pyplot as plt
from paraRange import *
# 训练模型
def trainModel(scale_flag):
    x_size, feature_size = X_train.shape
    print('==== Current Feature Size:%d.' % feature_size)
    if scale_flag:
        mms = preprocessing.StandardScaler()
        X_train_scaled = mms.fit_transform(X_train)
        # X_test_scaled = mms.transform(X_test)
    else:
        X_train_scaled = X_train
    gsA = searchMethodFun().fit(X_train_scaled, y_train)
    bestScore = gsA.best_score_
    best_params = gsA.best_params_
    best_estimator = gsA.best_estimator_
    print('==== The best parame is %s(with the score:%f).' %(best_params, bestScore))
    return best_params, best_estimator  #best_params,estimator.model

# 画出分类器预测的分布图 (by Afei)
# def plotDist(x, y, path):
#     plt.style.use('bmh')
#     plt.hist(x, color='teal', bins=20, normed=True, alpha=0.6)
#     plt.hist(y, color='darkred', bins=20, normed=True, alpha=0.6)
#     plt.legend(['no risk', 'has risk'])
#     plt.savefig(path +'/dist.png', dpi=300, format='png')
#     plt.ylabel('frequency')
#     plt.xlabel('linear_svm result')