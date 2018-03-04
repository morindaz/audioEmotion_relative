from keras.models import Sequential
from src.loadData import *
model = Sequential()
from keras.layers import Dense, Activation

model.add(Dense(units=64, input_dim=100))
model.add(Activation('relu'))
model.add(Dense(units=10))
model.add(Activation('softmax'))
model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.SGD(lr=0.01, momentum=0.9, nesterov=True))
model.fit(X_train, y_train, epochs=5, batch_size=32)
loss_and_metrics = model.evaluate(X_test, y_test, batch_size=128)
classes = model.predict(X_test, batch_size=128)
print (classes)