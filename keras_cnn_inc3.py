
from time import time
import numpy as np
import tensorflow.keras as keras
import tensorflow as tf
import os
import json
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
import warnings
# Keras Core
from keras.layers.convolutional import MaxPooling2D, Convolution2D, AveragePooling2D
from keras.layers import Input, Dropout, Dense, Flatten, Activation
from keras.layers.normalization import BatchNormalization
from keras.layers.merge import concatenate

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



inc_model=InceptionV3(include_top=False,weights='imagenet',input_shape=input_shape)

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

weights_filename='bottleneck_features_and_weights/fc_inception_cats_dogs_250.hdf5'

x = Flatten()(inc_model.output)
x = Dense(64, activation='relu', name='dense_one')(x)
x = Dropout(0.5, name='dropout_one')(x)
x = Dense(64, activation='relu', name='dense_two')(x)
x = Dropout(0.5, name='dropout_two')(x)
top_model=Dense(2, activation='softmax', name='output')(x)
model = Model(input=inc_model.input, output=top_model)
model.load_weights(weights_filename, by_name=True)
for layer in inc_model.layers[:205]:
    layer.trainable = False
    
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])

model.fit_generator(
        train_generator,
        steps_per_epoch=2000,
        epochs=200,
        validation_data=validation_generator,
        nb_val_samples=2000)