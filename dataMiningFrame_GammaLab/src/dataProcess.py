import numpy as np
import pandas as pd
testPath = '../data/test.csv'
trainPath = '../data/train2.csv'
df_test = pd.read_csv(testPath, sep=',', header=0)
# data = df.values[:, 0:-1]
data_test = df_test.values
df_train = pd.read_csv(trainPath,sep=',',header=0)
data_train = df_train.values
result = []
for i in data_train:
    if i in data_test:
        pass
        # print(i)
    else:
        result.append(i)
print(len(result))
result_data = pd.DataFrame(result)
result_data.to_csv("without_repeat.csv")