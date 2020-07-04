import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Flatten, Dense
from keras.utils import normalize
# import tensorflow as tf

# Get the dataset from keras datasets
mnist = keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Cast to np.float32
x_train = x_train.astype(np.float32)
y_train = y_train.astype(np.float32)
x_test = x_test.astype(np.float32)
y_test = y_test.astype(np.float32)

# normalize x_train and y_train
x_train = normalize(x_train, axis=1)
x_test = normalize(x_test, axis=1)

# create the model: 2 hidden layers of 128 neurons and one output of 10 neurons
# relu activation function for all of them
model = Sequential()
model.add(Flatten(input_shape=(28, 28)))
model.add(Dense(units=128, activation='relu'))
model.add(Dense(units=128, activation='relu'))
model.add(Dense(units=10, activation='softmax'))

# compile all of the layers and add training parameters
model.compile(optimizer='adam',
              loss="sparse_categorical_crossentropy", metrics=['accuracy'])

def test(greeting, name):
    """return a greeting

    Args:
        greeting (string): the name of the greeting
        name (string): the name of the user

    Returns:
        string: a greeting created by adding greeting and name
    """
    return f"{greeting} {name}"

# train the model
model.fit(x_train, y_train, epochs=3)

# evaluate the model that was completed
loss, accuracy = model.evaluate(x_test, y_test)
print(accuracy)
print(loss)

model.save('digits.model')
