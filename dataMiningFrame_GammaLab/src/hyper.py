#coding:utf-8
import xgboost as xgb
from sklearn.metrics import accuracy_score
from hyperopt import fmin, tpe, hp,space_eval,rand,Trials,partial,STATUS_OK
from loadData import *
from sklearn.metrics import precision_recall_fscore_support

def GBM(argsDict):
    # max_depth = argsDict["max_depth"]
    n_estimators = argsDict['n_estimators']
    learning_rate = argsDict["learning_rate"]
    # subsample = argsDict["subsample"]
    # min_child_weight = argsDict["min_child_weight"]

    params = {
        'booster': 'gbtree',
        #这里手写数字是0-9，是一个多类的问题，因此采用了multisoft多分类器，
        'objective': 'multi:softmax',
        # 'objective': 'multi:softprob',
        'num_class': 6,  # 类数，与 multisoftmax 并用
        'gamma': 0,  # 在树的叶子节点下一个分区的最小损失，越大算法模型越保守 。[0:] 不可以动
        'lambda': 100,
        'max_depth': 5,  # 构建树的深度 [1:]
        'subsample': 0.8,  # 采样训练数据，设置为0.5，随机选择一般的数据实例 (0:1]
        'colsample_bytree': 0.8,  # 构建树树时的采样比率 (0:1] 不可以动
        # 'min_child_weight':12, # 节点的最少特征数

        'silent': 1,
        'eta': hp.quniform('eta', 0.025, 0.5, 0.025),
        'seed': 710,
        'nthread': 1,  # cpu 线程数,根据自己U的个数适当调整
        'n_estimators': n_estimators,
        'learning_rate':learning_rate,
        'min_child_weight': 1
    }
    num_rounds = 5
    xg_train = xgb.DMatrix(X_train, label=y_train)
    xg_test = xgb.DMatrix(X_test, label=y_test)
    # setup parameters for xgboost
    watchlist = [(xg_train, 'train'), (xg_test, 'test')]
    plst = list(params.items())
    model_prob = xgb.train(plst,xg_train, num_rounds, watchlist, early_stopping_rounds=100)
    preds_prob = model_prob.predict(xg_test, ntree_limit=model_prob.best_iteration)
    acc = accuracy_score(y_test, preds_prob)
    fscore = precision_recall_fscore_support(y_test, preds_prob, average='macro')
    print (acc)
    print (fscore)
    return -acc

space = {
         # "max_depth":hp.choice("max_depth",range(2,20)),
         "n_estimators": hp.quniform('n_estimators', 10, 100, 1),  #[0,1,2,3,4,5] -> [50,]
         # "n_estimators":hp.choice("n_estimators",range(5,10)),  #[0,1,2,3,4,5] -> [50,]
         "learning_rate":hp.uniform("learning_rate", 0.05, 0.15), #[0,1,2,3,4,5] -> 0.05,0.06
         # "subsample":hp.uniform("subsample", 0.5, 0.9),#[0,1,2,3] -> [0.7,0.8,0.9,1.0]
         # "min_child_weight":hp.quniform('min_child_weight', 1, 10, 1), #
        }
algo = partial(tpe.suggest,n_startup_jobs=1)
best = fmin(GBM,space,algo=algo,max_evals=5)
print (best)
# best.save_model('./model/xgb.model')
print (GBM(best))