# dataMiningFrame_GammaLab
GammaLab通用的模型训练框架

###文件夹说明
dataMiningFrame_GammaLab
|-data：存放原始数据集
|-feature：存放筛选出来的特征结果
|-models：存放训练好的models参数 包括.pkl和.model
|-src：存放所有代码文件
    |-main.py程序入口 完整的过程，从开始筛选特征，到模型训练最后的结果输出
    |-dependence.py 路径配置文件，存放数据、特征等地址
    |-loadData.py 读入数据，依赖于dependence.py
    |-paraRange.py 模型相关参数，依赖于loadData.py
    |-featureSelection.py 特征选择功能，依赖于paraRange.py，可调整其中参数
    |-modelTrain.py 模型训练以及预测，依赖于paraRange.py

selectedCombinedAllwithLabel1是去噪后未筛选的音频+去噪后筛选的音频结合的结果
selectedCombinedAllwithLabel2是未去噪未筛选的音频+去噪后筛选的音频结合的结果
selectedCombinedAllwithLabel3是去噪后未筛选的音频+未去噪未筛选的音频+去噪后筛选的音频结合的结果

##分类依据 在audioEmotionNew里面
0\sadness (43,24,35,50,37,9,21,6,52,42,47,17,27,45)
1\fear(25,41,14,7,22)
2\surprise(5,48,32)
3\joy(33,34,36,1,26,18,46,53,51,0,28,44,11,8,40)
4\anger(10,16,13,15,29,2,3)
5\disgust(20,30,4,23,49,39,19,38,12,31)

[6, 9, 17, 21, 24, 27, 35, 37, 42, 43, 45, 47, 50, 52]
[7, 14, 22, 25, 41]
[5, 32, 48]
[0, 1, 8, 11, 18, 26, 28, 33, 34, 36, 40, 44, 46, 51, 53]
[2, 3, 10, 13, 15, 16, 29]
[4, 12, 19, 20, 23, 30, 31, 38, 39, 49]
(neutral)属于一个大类，下面还没有数据。persiveness现在还有，之后会删除

 参考的链接：
 1、bobo给的
 https://github.com/morindaz/Kaggle_CrowdFlower
 2、整合几种方法的李子
 http://scikit-learn.org/stable/auto_examples/ensemble/plot_feature_transformation.html#example-ensemble-plot-feature-transformation-py
 3、GBDT使用方法
 https://github.com/morindaz/GBDT
 4、GBDT调参
 http://chuansong.me/n/296022746725
 5、XGBoost调参
 https://www.dataiku.com/learn/guide/code/python/advanced-xgboost-tuning.html
 https://github.com/bamine/Kaggle-stuff/blob/master/otto/hyperopt_xgboost.py
 http://blog.csdn.net/qq_34139222/article/details/60322995
 6、Hyperopt+randomSearch
 https://stats.stackexchange.com/questions/183984/how-to-use-xgboost-cv-with-hyperparameters-optimization
 
 _______________________________
 