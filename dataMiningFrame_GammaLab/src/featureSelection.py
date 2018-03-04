# coding=utf-8
'''
此文件完成特征选择的功能。
从dependence中加载对应的特征选择方法，作为featureSelect(selectionMethod)中
selectionMethod的参数，此处为SFS
'''
from paraRange import *
#######################featureSelection相关参数定义
#selectionMethod需要的分类器定义，此处为LogisticRegression
def selectionMethodFun():
    classifier = LogisticRegression(C=0.5,penalty='l2',multi_class='multinomial',class_weight='balanced',solver='lbfgs')
    #selectionMethod需要的特征选择方法，此处为SFS。需要被加载到Main函数中
    selectionMethod = SFS(classifier,k_features=5,forward=True,floating=False,verbose=2,cv=2,scoring='accuracy')
    return selectionMethod

def featureSelect():
       #通过从dependence文件中加载的selectionMethod方法
       selected = selectionMethodFun().fit(data, Y)
       a = selected.subsets_
       avalue = list(a.values())
       indexVal = avalue[-1]['feature_idx']
       #依次将获取到的结果存储到featureIdx中
       featureIdx = []
       for item in indexVal:
              featureIdx.append(item)
       print featureIdx
       #将featureIdx的结果导出到feature路径下面
       outPut = {'Index': featureIdx}
       output_Archive = pd.DataFrame(outPut)
       output_Archive.to_csv(featureBasic+'selectedMorindaz.csv')  #保存特征到对应的路径


if __name__ == "__main__":
       featureSelect()
