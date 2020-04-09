#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CNN used for training and classification of digits in the Cifar-10 dataset provided by Keras
@author: simsshepherd
"""

from __future__ import print_function
import keras
from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator

batch_size = 32
num_classes = 10
epochs = 25

# Get data
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# Convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

## Create augmented image generator
#datagen = ImageDataGenerator(horizontal_flip=True, 
#                             rotation_range=90, 
#                             featurewise_std_normalization=True, 
#                             width_shift_range=.1, 
#                             height_shift_range=.1)
## fit the training images to the augmented image generator
#datagen.fit(x_train)

model = Sequential()
# Add same padding to edges so that we do not lose as much data in those positions
model.add(Conv2D(32, (3, 3), padding='same'))  
# Relu chosen to create high training speed along with better performance than sigmoid functions
model.add(Activation('relu')) 
model.add(Conv2D(32, (3, 3), padding='same'))
model.add(Activation('relu'))
# Pass through pooling layer to downsample the features 
model.add(MaxPooling2D(pool_size=(2, 2)))
# Dropout of .25 to regularize data and preent overfitting
model.add(Dropout(0.25))

model.add(Conv2D(32, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(Conv2D(32, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(32, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(Conv2D(32, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

# Flatten the output and feed into final Dense layer to sonsolidate into the 10
# potential class outputs, then use softmax to choose the one with maximum likelihood
model.add(Flatten())
model.add(Dense(num_classes))
model.add(Activation('softmax'))

# Train the model using categorical_crossentropy for the loss, an adam optimizer, and report accuracy metrics
model.compile(loss='categorical_crossentropy', optimizer="adam", metrics=['accuracy'])

## Fit augmented generator images to model ### NOT WORKING ###
#fitted = model.fit_generator(datagen.flow(x_train, y_train, batch_size=32), 
#                             steps_per_epoch=len(x_train)/32, 
#                             verbose = 1, 
#                             epochs=epochs, 
#                             validation_data=(x_test,y_test))

# Fit the data to the model and run with selected batch size and epochs
# Report validation testing at the end of ecah epoch 
# by providing the validation data and setting verbose = 1
fitted = model.fit(x_train, y_train, batch_size=batch_size, epochs = epochs, 
                   validation_data=(x_test, y_test), verbose=1)

# Score trained model
scores = model.evaluate(x_test, y_test, verbose=1)
print('Test loss:', scores[0])
print('Test accuracy:', scores[1])
