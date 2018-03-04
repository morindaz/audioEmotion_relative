import keras
print(keras.__version__)
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import SGD
from loadData import *
from keras.models import Sequential
from keras.layers import Dense,LSTM,Embedding
model = Sequential()
y_test =keras.utils.to_categorical(y_test)
y_train = keras.utils.to_categorical(y_train)
print (X_train.shape)
print(y_train)
# model.add(Dense(664, activation='relu', input_dim=978))
# model.add(Dropout(0.4))
model.add(Dense(464, activation='relu', input_dim=978))
# model.add(Embedding(978, 4))
# model.add(LSTM(16))
model.add(Dense(82, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(54, activation='softmax'))


sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy',optimizer=sgd,metrics=['accuracy'])

model.fit(X_train, y_train,epochs=20,batch_size=128)
score = model.evaluate(X_test, y_test, batch_size=128)
# scores = model.evaluate(X, Y)
print("\n%s: %.2f%%" % (model.metrics_names[1], score[1]*100))