# -*- coding: utf-8 -*-
# Part 1 - Building the CNN

# Importing the Keras libraries and packages
from keras.models import Sequential#to initiaise neural network
from keras.layers import Conv2D#to add conv layers
from keras.layers import MaxPooling2D# for pooling layers
from keras.layers import Flatten# flattening
from keras.layers import Dense#add fully connected layers 

# Initialising the CNN
classifier = Sequential()
# Step 1 - Convolution
classifier.add(Conv2D(32, (3, 3), input_shape = (64, 64, 3), activation = 'relu'))
'''32 feature detectors, 3*3 matrix of feature detector, input image--convert all imgs to one single size--useing 3 channels also but tensor flow format as using tensor backend not theano ---using less values or else run code with gpu for 8hrs
using rectifier fun for non linearity'''
 
# Step 2 - Pooling--to avoid time complexity and less computation
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Adding a second convolutional layer # to imporve accuracy also increase target size for more accuracy as more data so more accuracy
classifier.add(Conv2D(32, (3, 3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Step 3 - Flattening
classifier.add(Flatten())
 # Step 4 - Full connection
classifier.add(Dense(units = 128, activation = 'relu'))
classifier.add(Dense(units = 1, activation = 'sigmoid'))
# Compiling the CNN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Part 2 - Fitting the CNN to the images

from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale = 1./255, #rescale values # 0-1 from 0-255
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)

training_set = train_datagen.flow_from_directory('dataset/training_set',
                                                 target_size = (64, 64),#dimensions of cnn # increase target size for more accuracy as more data so more accuracy
                                                 batch_size = 32,
                                                 class_mode = 'binary') #its binary dep var

test_set = test_datagen.flow_from_directory('dataset/test_set',
                                            target_size = (64, 64),#increase target size for more accuracy as more data so more accuracy
                                            batch_size = 32,
                                            class_mode = 'binary')

classifier.fit_generator(training_set,
                         steps_per_epoch = 8000,# images in training set
                         epochs = 25,
                         validation_data = test_set,
                         validation_steps = 2000)
