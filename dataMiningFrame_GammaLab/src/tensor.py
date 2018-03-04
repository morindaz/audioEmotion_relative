import numpy as np
from src.loadData import *
# from keras.models import Sequential
# from keras.layers import Dense, Dropout
# print (y_train)
# # print (X_test)
# print (1111)
# model = Sequential()
# model.add(Dense(64, input_dim=30, activation='relu'))
# model.add(Dropout(0.5))
# model.add(Dense(64, activation='relu'))
# model.add(Dropout(0.5))
# model.add(Dense(1, activation='sigmoid'))
#
# model.compile(loss='binary_crossentropy',
#               optimizer='rmsprop',
#               metrics=['accuracy'])
# model.fit(X_train, y_train,
#           epochs=20,
#           batch_size=128)
# score = model.evaluate(X_test, y_test, batch_size=128)
# print (score)

import numpy
import pandas
from keras.models import Sequential
# from keras.layers import Dense
from keras.layers import Masking
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from keras.layers import Dense, Dropout
from keras.layers import Embedding
from keras.layers import LSTM
from sklearn.pipeline import Pipeline

from sklearn import datasets
# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)

# encode class values as integers
encoder = LabelEncoder()
encoder.fit(y_train)
encoded_Y = encoder.transform(y_train)
# convert integers to dummy variables (i.e. one hot encoded)
dummy_y = np_utils.to_categorical(encoded_Y)

# define baseline model
def baseline_model():
	model = Sequential()
	model.add(Dense(8, input_dim=980, activation='relu'))
	model.add(Dense(6, activation='softmax'))
	# Compile model
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	return model


estimator = KerasClassifier(build_fn=baseline_model, epochs=200, batch_size=5, verbose=0)
kfold = KFold(n_splits=10, shuffle=True, random_state=seed)
results = cross_val_score(estimator, X_train, dummy_y, cv=kfold)
print("Baseline: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))