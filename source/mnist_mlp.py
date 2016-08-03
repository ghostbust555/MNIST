'''Trains a simple deep NN on the MNIST dataset.
Gets to 98.34% test accuracy after 20 epochs
'''

from __future__ import print_function
import numpy as np
import PIL
from scipy.misc import imread

np.random.seed(1337)  # for reproducibility

from keras.datasets import mnist
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import SGD, Adam, RMSprop
from keras.utils import np_utils


# number of samples to group before a weight update
batch_size = 256


nb_classes = 10
nb_epoch = 20

# the data, shuffled and split between train and test sets
(input_train, output_train), (input_test, output_test) = mnist.load_data()

# convert 28x28 pixel 2d array to 764 pixel 1d array, (append pixel from top left to bottom right)
input_train = input_train.reshape(60000, 784)
input_test = input_test.reshape(10000, 784)

input_train = input_train.astype('float32')
input_test = input_test.astype('float32')

# convert pixels from 0-255 to 0-1
input_train /= 255
input_test /= 255

print(input_train.shape[0], 'train samples')
print(input_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
# input looks like 0, 7, 4...
# output looks like [1,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,1,0,0],[0,0,0,0,1,0,0,0,0,0] ...
output_train = np_utils.to_categorical(output_train, nb_classes)
output_test = np_utils.to_categorical(output_test, nb_classes)

# initialize a sequential network
model = Sequential()

# Layer 1 (input layer)
model.add(Dense(512, input_shape=(784,)))
model.add(Activation("relu"))
model.add(Dropout(0.35))

# Layer 2
model.add(Dense(256))
model.add(Activation('relu'))
model.add(Dropout(0.35))

# Layer 3 (output layer)
model.add(Dense(10))
model.add(Activation('softmax'))

model.summary()

model.compile(loss='mse',
              optimizer=RMSprop(),
              metrics=['accuracy'])

history = model.fit(input_train, output_train,
                    batch_size=batch_size, nb_epoch=nb_epoch,
                    verbose=1, validation_data=(input_test, output_test))
score = model.evaluate(input_test, output_test, verbose=0)
print('Test score:', score[0])
print('Test accuracy:', score[1])

# Grab my test image for verification that everything is working well
zero = imread("0.png").astype('float32') / 255
zero = zero.reshape(1, 784)

print("\n\n classification of image 0.png = {0}".format(model.predict_classes(zero)))
