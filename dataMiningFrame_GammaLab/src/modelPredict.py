# coding=utf-8
from __future__ import division
from modelTrain import *
import os
from sklearn.externals import joblib
from sklearn.metrics import accuracy_score
import pyttsx
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn import metrics
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import (RandomTreesEmbedding, RandomForestClassifier,
                              GradientBoostingClassifier)
from sklearn.preprocessing import OneHotEncoder

# from xgboost import *
from sklearn.neural_network import MLPClassifier
def predict():
    # feature_flag = False
    param_flag = False
    myfile = os.path.exists(modelPath+paramName)
    if myfile:
        param_flag = True
        print('==== 参数最优化已经存在.')

    if not param_flag:
        scaled_flag = False
        best_params, model = trainModel(scaled_flag)
        # 保存模型参数
        joblib.dump(best_params,modelPath+paramName)
        joblib.dump(model, modelPath+estimatorName)
    param = None
    # 从文件读取LinearSVC最佳参数设定
    clf = joblib.load(modelPath+estimatorName)
    # clf = MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(9, 2), random_state=1)
    n_estimator = 60

    # clf = GradientBoostingClassifier(n_estimators=n_estimator,learning_rate=0.05, max_depth=4, min_samples_leaf=17)

    # clf = LogisticRegression(C=0.8,penalty='l2',multi_class='multinomial',class_weight='balanced',solver='lbfgs')
    # clf =LinearSVC(C=1.8)
    # clf =MultinomialNB()
    # clf =RandomForestClassifier(max_depth=2, random_state=0)

    param = joblib.load(modelPath+paramName)
    # linearsvc classifier
    print(clf)
    print (param)
    # stddc = clf.named_steps['standardscaler']
    # model = clf.named_steps[modelName]
    # print(model.coef_)
    # 样本外测试
    # clf = MLPClassifier()
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print (y_pred)
    confusion_matrix = metrics.confusion_matrix(y_test, y_pred)
    prob = confusion_matrix
    sorted = []
    for i in confusion_matrix:
        b = i.argsort()[::-1]
        sorted.append(b)
    print('==== The Confuse-Matrix is:%s' % confusion_matrix)
    testData = pd.DataFrame(confusion_matrix)
    testData.to_csv("confusion_matrix.csv")
    sortedData = pd.DataFrame(sorted)
    sortedData.to_csv("sortedData.csv")

    result = []
    for i in range(len(prob)):
        print (prob[i])
        res = []
        all = sum(prob[i])
        print(all)
        for j in range(len(prob[i])):
            a = prob[i][j] / all
            print(a)
            res.append(a)
        # print prob[i]
        result.append(res)
    # print prob
    # print 1/25
    for i in range(len(result)):
        result[i].sort(reverse = True)


    prob_matrix = pd.DataFrame(result)
    prob_matrix.to_csv("prob_matrix.csv")


    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    f11 = metrics.precision_score(y_test, y_pred, labels=[0, 1, 2, 3,4,5], average='macro')
    print('==== The Accuracy is:%s' % acc)
    print('==== The F1_score is:%s' % f1)
    print('==== The F11_score is:%s' % f11)
    # print(model.intercept_)
    # engine = pyttsx.init()
    # engine.say('Congratulations!')
    # engine.runAndWait()


if __name__ =='__main__':
    predict()