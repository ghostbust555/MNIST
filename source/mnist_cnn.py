'''Trains a simple convnet on the MNIST dataset.
Gets to 96.80% test accuracy after 12 epochs
(there is still a lot of margin for parameter tuning).
'''

from __future__ import print_function
import numpy as np
np.random.seed(1337)  # for reproducibility

from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils


batch_size = 128
nb_classes = 10
nb_epoch = 20

# input image dimensions
img_rows, img_cols = 28, 28
# number of convolutional filters to use
nb_filters = 1
# size of pooling area for max pooling
nb_pool = 2
# convolution kernel size
nb_conv = 3

# the data, shuffled and split between train and test sets
(input_train, output_train), (input_test, output_test) = mnist.load_data()

input_train = input_train.reshape(input_train.shape[0], 1, img_rows, img_cols)
input_test = input_test.reshape(input_test.shape[0], 1, img_rows, img_cols)

input_train = input_train.astype('float32')
input_test = input_test.astype('float32')

# convert pixels from 0-255 to 0-1
input_train /= 255
input_test /= 255

print('X_train shape:', input_train.shape)
print(input_train.shape[0], 'train samples')
print(input_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
# input looks like 0, 7, 4...
# output looks like [1,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,1,0,0],[0,0,0,0,1,0,0,0,0,0] ...
Y_train = np_utils.to_categorical(output_train, nb_classes)
Y_test = np_utils.to_categorical(output_test, nb_classes)

# initialize a sequential network
model = Sequential()

model.add(Convolution2D(nb_filters, nb_conv, nb_conv,
                        border_mode='valid',
                        input_shape=(1, img_rows, img_cols)))
model.add(Activation('relu'))
model.add(Convolution2D(nb_filters, nb_conv, nb_conv))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool)))
model.add(Dropout(0.25))

# take the 2d arrays of neurons and flatten them into a list of neurons
model.add(Flatten())

model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(nb_classes))
model.add(Activation('softmax'))

model.summary()

model.compile(loss='categorical_crossentropy',
              optimizer='adadelta',
              metrics=['accuracy'])

model.fit(input_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch,
          verbose=1, validation_data=(input_test, Y_test))
score = model.evaluate(input_test, Y_test, verbose=0)
print('Test score:', score[0])
print('Test accuracy:', score[1])
