import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Flatten, Dense, Conv2D, MaxPooling2D
from keras.utils import normalize
import tensorflow as tf

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.compat.v1.Session(config=config)

# Get the dataset from keras datasets
mnist = keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Cast to np.float32
x_train = x_train.astype(np.float32)
y_train = y_train.astype(np.float32)
x_test = x_test.astype(np.float32)
y_test = y_test.astype(np.float32)

print(x_train.shape)
# normalize x_train and y_train
x_train = normalize(x_train, axis=1)
x_test = normalize(x_test, axis=1)

x_train = x_train[..., np.newaxis]
x_test = x_test[..., np.newaxis]

# create the model: 2 hidden layers of 128 neurons and one output of 10 neurons
# relu activation function for all of them
model = Sequential()
model.add(Conv2D(4, (3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(MaxPooling2D((2, 2)))
model.add(Flatten())
model.add(Dense(units=128, activation='relu'))
model.add(Dense(units=128, activation='relu'))
model.add(Dense(units=10, activation='softmax'))

model.summary()

# compile all of the layers and add training parameters
model.compile(optimizer='adam',
              loss="sparse_categorical_crossentropy", metrics=['accuracy'])

# train the model
model.fit(x_train, y_train, epochs=10)

# evaluate the model that was completed
loss, accuracy = model.evaluate(x_test, y_test)
print(accuracy)
print(loss)

model.save('digits.model')
