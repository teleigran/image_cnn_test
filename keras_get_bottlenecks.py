#!/usr/bin/env python3
from time import time
import numpy as np
import tensorflow.keras as keras
import tensorflow as tf
import os
import json
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
import warnings
import h5py
# Keras Core
from keras.layers.convolutional import MaxPooling2D, Convolution2D, AveragePooling2D
from keras.layers import Input, Dropout, Dense, Flatten, Activation
from keras.layers.normalization import BatchNormalization
from keras.models import Sequential

from keras.applications.inception_v3 import InceptionV3
from keras import regularizers
from keras import initializers
from keras.models import Model

from keras import backend as K
from keras.callbacks import TensorBoard

physical_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

# dimensions of our images.
img_width, img_height = 128, 128

train_data_dir = 'data/train'
validation_data_dir = 'data/validation'
nb_train_samples = 12000
nb_validation_samples = 1240
epochs = 20
batch_size =8

i=0
f=open("image_cnn_inception.labels","w")
for name in os.listdir(train_data_dir):
    ##if os.path.isdir(name):
    print(str(i)+" "+name)
    f.write(str({i : name })+"\n")
    i=i+1

f.close()
        
if K.image_data_format() == 'channels_first':
    input_shape = (3, img_width, img_height)
else:
    input_shape = (img_width, img_height, 3)



model=InceptionV3(include_top=False,weights='imagenet',input_shape=input_shape)

# this is the augmentation configuration we will use for training
train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

# this is the augmentation configuration we will use for testing:
# only rescaling
test_datagen = ImageDataGenerator(rescale=1. / 255)

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical')

validation_generator = test_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical')

tensorboard = TensorBoard(log_dir='/tmp/tflogs/{}'.format(time()))

bottleneck_features_train = model.predict_generator(train_generator, 2000)
np.save(open('image_bn_features_train.npy', 'wb'), bottleneck_features_train)

bottleneck_features_validation = model.predict_generator(validation_generator, 2000)
np.save(open('image_bn_features_validation.npy', 'wb'), bottleneck_features_validation)

train_data = np.load(open('image_bn_features_train.npy', 'rb'))
train_labels = np.array([0] * 1000 + [1] * 1000) 

validation_data = np.load(open('image_bn_features_validation.npy', 'rb'))
validation_labels = np.array([0] * 1000 + [1] * 1000)

fc_model = Sequential()
fc_model.add(Flatten(input_shape=train_data.shape[1:]))
fc_model.add(Dense(64, activation='relu', name='dense_one'))
fc_model.add(Dropout(0.5, name='dropout_one'))
fc_model.add(Dense(64, activation='relu', name='dense_two'))
fc_model.add(Dropout(0.5, name='dropout_two'))
fc_model.add(Dense(1, activation='sigmoid', name='output'))

fc_model.compile(optimizer='rmsprop', 
              loss='categorical_crossentropy', 
              metrics=['accuracy'])

fc_model.fit(train_data, train_labels,
            nb_epoch=50, batch_size=32,
            validation_data=(validation_data, validation_labels))

fc_model.save_weights('fc_inception_v3.hdf5')