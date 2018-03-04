selectedCombinedAllwithLabel1是去噪后未筛选的音频+去噪后筛选的音频结合的结果
selectedCombinedAllwithLabel2是未去噪未筛选的音频+去噪后筛选的音频结合的结果
selectedCombinedAllwithLabel3是去噪后未筛选的音频+未去噪未筛选的音频+去噪后筛选的音频结合的结果
==== The Confuse-Matrix is:<function confusion_matrix at 0x0000000005B8B588>
==== The Accuracy is:0.387375113533
==== The F1_score is:0.36607923366
==== The F11_score is:0.327589669494


XGBOOST（1）
==== The Accuracy is:0.658492279746

（2）
==== CSV数据已经存在.
==== The Accuracy is:0.375113533152

（3）
==== CSV数据已经存在.
==== The Accuracy is:0.505548037889

linearSVM 2289行数据 30个特征 5大类  0.338568935428

xgBoost [366]   train-merror:0.050377   test-merror:0.657264
调整参数class=10
==== The Accuracy is:0.341325811001
==== The F1_score is:0.288724335674
==== The F11_score is:0.372562895669

class = 56
==== The Accuracy is:0.341325811001
==== The F1_score is:0.283538266192
==== The F11_score is:0.441121403336

max_depth:8 n_estimator:60 learning_rate:0.09 subsample:1.0 min_child_weight:3
==== The Accuracy is:0.370944992948
==== The F1_score is:0.336756858327
==== The F11_score is:0.342996132786

在以上参数 'num_class':20, 
==== The Accuracy is:0.377997179126
==== The F1_score is:0.349644860881
==== The F11_score is:0.313871654126

max_depth:17,n_estimator:55,learning_rate:0.15,subsample:0.8,min_child_weight:3
==== The Accuracy is:0.373765867419
==== The F1_score is:0.349914835433
==== The F11_score is:0.345502069476


-----------------------------------
gradientBoost 调整estimators的结果
estimators 10
==== The Accuracy is:0.32581100141
==== The F1_score is:0.261415188686
==== The F11_score is:0.17009075311
-----------------------------------
20
==== The Accuracy is:0.337094499295
==== The F1_score is:0.280993346849
==== The F11_score is:0.201808177929

60
==== The Accuracy is:0.335684062059
==== The F1_score is:0.297629580757
==== The F11_score is:0.212802086434

100  tuned
==== The Accuracy is:0.337094499295
==== The F1_score is:0.284534970726
==== The F11_score is:0.210525564269

500 
==== The Accuracy is:0.322990126939
==== The F1_score is:0.29810696564
==== The F11_score is:0.30379622532


linearSVC
feature 30  0.262341325811
feature 26  0.263751763047
feature 23  0.251057827927

logisticRegression C=0.5,penalty='l2',multi_class='multinomial',class_weight='balanced',solver='lbfgs'
acc 0.225669957687

linearSVC 
C=0.5  0.324400564175
C=0.1  0.331452750353
==== The Accuracy is:0.331452750353
==== The F1_score is:0.270045023202
==== The F11_score is:0.21510681982


randomForest
==== The Accuracy is:0.32581100141
==== The F1_score is:0.225845891493
==== The F11_score is:0.114995541288

 
计算在xgboost下面的top2 accuracy
=========This is:top2
401
709
0.565585331453
=========This is:top3
539
709
0.760225669958
=========This is:top4
630
709
0.888575458392
=========This is:top5
677
709
0.954866008463
=========This is:top6
709
709
1.0

54个分类的情况

=========This is:top2
93
709
0.131170662906
=========This is:top3
121
709
0.170662905501
=========This is:top4
139
709
0.19605077574
=========This is:top5
154
709
0.217207334274
=========This is:top6
164
709
0.231311706629