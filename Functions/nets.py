
import numpy as np
import random

#Keras
from keras.models import Model
from keras.layers import Input, concatenate, Conv2D, MaxPooling2D, UpSampling2D, Reshape, core, Dropout
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras import backend as K
from keras.optimizers import SGD
from keras import regularizers
from keras.optimizers import Adam, SGD, Adagrad
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Activation, Input
from keras.layers import Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.utils import np_utils
from keras.callbacks import EarlyStopping
from keras.utils import plot_model
from keras.models import model_from_json
from PIL import Image
import numpy as np
from matplotlib import pyplot as plt

def get_unet(n_ch,patch_height,patch_width):

   inputs = Input(shape=(n_ch,patch_height,patch_width))
   conv1 = Conv2D(32, (3, 3), activation='relu', padding='same',data_format='channels_first')(inputs)
   conv1 = Dropout(0.2)(conv1)
   conv1 = Conv2D(32, (3, 3), activation='relu', padding='same',data_format='channels_first')(conv1)
   pool1 = MaxPooling2D((2, 2))(conv1)
   #
   conv2 = Conv2D(64, (3, 3), activation='relu', padding='same',data_format='channels_first')(pool1)
   conv2 = Dropout(0.2)(conv2)
   conv2 = Conv2D(64, (3, 3), activation='relu', padding='same',data_format='channels_first')(conv2)
   pool2 = MaxPooling2D((2, 2))(conv2)
   #
   conv3 = Conv2D(128, (3, 3), activation='relu', padding='same',data_format='channels_first')(pool2)
   conv3 = Dropout(0.2)(conv3)
   conv3 = Conv2D(128, (3, 3), activation='relu', padding='same',data_format='channels_first')(conv3)

   up1 = UpSampling2D(size=(2, 2))(conv3)
   up1 = concatenate([conv2,up1],axis=1)
   conv4 = Conv2D(64, (3, 3), activation='relu', padding='same',data_format='channels_first')(up1)
   conv4 = Dropout(0.2)(conv4)
   conv4 = Conv2D(64, (3, 3), activation='relu', padding='same',data_format='channels_first')(conv4)
   #
   up2 = UpSampling2D(size=(2, 2))(conv4)
   up2 = concatenate([conv1,up2], axis=1)
   conv5 = Conv2D(32, (3, 3), activation='relu', padding='same',data_format='channels_first')(up2)
   conv5 = Dropout(0.2)(conv5)
   conv5 = Conv2D(32, (3, 3), activation='relu', padding='same',data_format='channels_first')(conv5)
   #
   conv6 = Conv2D(2, (1, 1), activation='relu',padding='same',data_format='channels_first')(conv5)
   conv6 = core.Reshape((2,patch_height*patch_width))(conv6)
   conv6 = core.Permute((2,1))(conv6)
   ############
   conv7 = core.Activation('softmax')(conv6)

   model = Model(input=inputs, output=conv7)

   # sgd = SGD(lr=0.01, decay=1e-6, momentum=0.3, nesterov=False)
   model.compile(optimizer='sgd', loss='categorical_crossentropy',metrics=['accuracy'])

   return model
def get_ClassicNet(n_ch,patch_size):
    model = Sequential()
    model.add(Conv2D(32,4,data_format = "channels_first",activation=None,
                                                      padding='same',
                                                      input_shape=(1,patch_size,patch_size),
                                                      kernel_initializer='random_normal',
                                                      bias_initializer='zeros',
                                                      kernel_regularizer=regularizers.l2(1e-5),
                                                      bias_regularizer = regularizers.l2(1e-5),name='C01'))     
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=None,  name='MP01',data_format = "channels_first"))
    model.add(Conv2D(32,4,data_format = "channels_first",activation=None,
                                                      padding='same',
                                                      kernel_initializer='random_normal',
                                                      bias_initializer='zeros',
                                                      kernel_regularizer=regularizers.l2(1e-5),
                                                      bias_regularizer = regularizers.l2(1e-5),name='C02'))

    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=None,  name='MP02',data_format = "channels_first"))
    model.add(Conv2D(64,4,data_format = "channels_first",activation=None,
                                                      padding='same',
                                                      kernel_initializer='random_normal',
                                                      bias_initializer='zeros',
                                                      kernel_regularizer=regularizers.l2(1e-5),
                                                      bias_regularizer = regularizers.l2(1e-5),name='C03'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=None,  name='MP03',data_format = "channels_first"))
    model.add(Conv2D(64,4,data_format = "channels_first",activation=None,
                                                      padding='same',
                                                      kernel_initializer='random_normal',
                                                      bias_initializer='zeros',
                                                      kernel_regularizer=regularizers.l2(1e-5),
                                                      bias_regularizer = regularizers.l2(1e-5),name='C04'))
    model.add(Activation('relu'))
    model.add(Conv2D(128,4,data_format = "channels_first",activation=None,
                                                      padding='same',
                                                      kernel_initializer='random_normal',
                                                      bias_initializer='zeros',
                                                      kernel_regularizer=regularizers.l2(1e-5),
                                                      bias_regularizer = regularizers.l2(1e-5),name='C05'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=None,  name='MP04',data_format = "channels_first"))
    model.add(Flatten())
    model.add(Dense(100, kernel_initializer='random_normal', 
                                                bias_initializer='zeros',
                                                kernel_regularizer=regularizers.l2(1e-5),
                                                bias_regularizer = regularizers.l2(1e-5)))
    model.add(Dense(1, kernel_initializer='random_normal', 
                                                bias_initializer='zeros',
                                                kernel_regularizer=regularizers.l2(1e-5)))
    model.compile(loss='binary_crossentropy',
    optimizer=Adam(lr=1e-3, beta_1=0.9, beta_2=0.999, epsilon=1e-5, decay=0),
    metrics=['accuracy'])      
  
    return model
def get_Liskowski(patch_height,patch_width):
    model = Sequential()
    model.add(Conv2D(64,3,data_format = "channels_first",activation=None,
                                                      padding='same',
                                                      input_shape=(1,patch_height,patch_width),
                                                      kernel_initializer='random_normal',
                                                      bias_initializer='zeros',
                                                      kernel_regularizer=regularizers.l2(1e-5),
                                                      bias_regularizer = regularizers.l2(1e-5),name='C01'))
    model.add(Activation('relu'))
    #model.add(BatchNormalization())
    model.add(Dropout(0.2))
    model.add(Conv2D(64,3,data_format = "channels_first",
                                                      padding='valid',
                                                      kernel_initializer='random_normal',
                                                      bias_initializer='zeros',
                                                      kernel_regularizer=regularizers.l2(1e-5),
                                                      bias_regularizer = regularizers.l2(1e-5),   name='C02'))
    model.add(Activation('relu'))
    #model.add(BatchNormalization())
    model.add(Conv2D(128,3,data_format = "channels_first",
                                                      padding='valid',
                                                      kernel_initializer='random_normal',
                                                      bias_initializer='zeros',
                                                      kernel_regularizer=regularizers.l2(1e-5),
                                                      bias_regularizer = regularizers.l2(1e-5),   name='C03'))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))
    #model.add(BatchNormalization())
    model.add(Conv2D(128,3,data_format = "channels_first",
                                                      padding='valid',
                                                      kernel_initializer='random_normal',
                                                      bias_initializer='zeros',
                                                      kernel_regularizer=regularizers.l2(1e-5),
                                                      bias_regularizer = regularizers.l2(1e-5),   name='C04'))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))
    #model.add(BatchNormalization())
    model.add(Conv2D(128,3,data_format = "channels_first",
                                                      padding='valid',
                                                      kernel_initializer='random_normal',
                                                      bias_initializer='zeros',
                                                      kernel_regularizer=regularizers.l2(1e-5),
                                                      bias_regularizer = regularizers.l2(1e-5),   name='C05'))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))
    #model.add(BatchNormalization())
    model.add(Flatten())
    model.add(Dense(512, kernel_initializer='random_normal', 
                                                bias_initializer='zeros',
                                                kernel_regularizer=regularizers.l2(1e-5),
                                                bias_regularizer = regularizers.l2(1e-5)))
    model.add(Dense(512, kernel_initializer='random_normal', 
                                                bias_initializer='zeros',
                                                kernel_regularizer=regularizers.l2(1e-5),
                                                bias_regularizer = regularizers.l2(1e-5)))
  
    model.add(Dense(1, kernel_initializer='random_normal', 
                                                bias_initializer='zeros',
                                                kernel_regularizer=regularizers.l2(1e-5),
                                                bias_regularizer = regularizers.l2(1e-5)))

    model.compile(loss='binary_crossentropy',
    optimizer=Adam(lr=1e-3, beta_1=0.9, beta_2=0.999, epsilon=1e-5, decay=0),
    metrics=['accuracy'])      


    return model
def get_LiskowskiSP(patch_height,patch_width):
    model = Sequential()
    model.add(Conv2D(64,3,data_format = "channels_first",activation=None,
                                                      padding='same',
                                                      input_shape=(1,patch_height,patch_width),
                                                      kernel_initializer='random_normal',
                                                      bias_initializer='zeros',
                                                      kernel_regularizer=regularizers.l2(1e-5),
                                                      bias_regularizer = regularizers.l2(1e-5),name='C01'))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))
    model.add(Conv2D(64,3,data_format = "channels_first",
                                                      padding='valid',
                                                      kernel_initializer='random_normal',
                                                      bias_initializer='zeros',
                                                      kernel_regularizer=regularizers.l2(1e-5),
                                                      bias_regularizer = regularizers.l2(1e-5),   name='C02'))
    model.add(Activation('relu'))
    model.add(Conv2D(128,3,data_format = "channels_first",
                                                      padding='valid',
                                                      kernel_initializer='random_normal',
                                                      bias_initializer='zeros',
                                                      kernel_regularizer=regularizers.l2(1e-5),
                                                      bias_regularizer = regularizers.l2(1e-5),   name='C03'))
    model.add(Activation('relu'))
    model.add(Conv2D(128,3,data_format = "channels_first",
                                                      padding='valid',
                                                      kernel_initializer='random_normal',
                                                      bias_initializer='zeros',
                                                      kernel_regularizer=regularizers.l2(1e-5),
                                                      bias_regularizer = regularizers.l2(1e-5),   name='C04'))
    model.add(Activation('relu'))
    model.add(Conv2D(128,3,data_format = "channels_first",
                                                      padding='valid',
                                                      kernel_initializer='random_normal',
                                                      bias_initializer='zeros',
                                                      kernel_regularizer=regularizers.l2(1e-5),
                                                      bias_regularizer = regularizers.l2(1e-5),   name='C05'))
    model.add(Activation('relu'))
    model.add(Flatten())
    model.add(Dense(512, kernel_initializer='random_normal', 
                                                bias_initializer='zeros',
                                                kernel_regularizer=regularizers.l2(1e-5),
                                                bias_regularizer = regularizers.l2(1e-5)))
    model.add(Activation('relu'))
    model.add(Dense(512, kernel_initializer='random_normal', 
                                                bias_initializer='zeros',
                                                kernel_regularizer=regularizers.l2(1e-5),
                                                bias_regularizer = regularizers.l2(1e-5)))
    model.add(Activation('relu'))
    model.add(Dense(9, activation='softmax'))


    model.compile(loss='binary_crossentropy',
    optimizer=Adam(lr=1e-3, beta_1=0.9, beta_2=0.999, epsilon=1e-5, decay=0),
    metrics=['accuracy'])            


    return model
def get_unet02(n_ch,patch_height,patch_width):
    model = Sequential()
    model.add(Conv2D(32,3,data_format = "channels_first",activation=None,
                                                      padding='same',
                                                      input_shape=(1,patch_height,patch_width),
                                                      kernel_initializer='random_normal',
                                                      bias_initializer='zeros',
                                                      kernel_regularizer=regularizers.l2(1e-5),
                                                      bias_regularizer = regularizers.l2(1e-5),name='C01'))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))
    model.add(Conv2D(32,3,data_format = "channels_first",
                                                      padding='valid',
                                                      kernel_initializer='random_normal',
                                                      bias_initializer='zeros',
                                                      kernel_regularizer=regularizers.l2(1e-5),
                                                      bias_regularizer = regularizers.l2(1e-5),   name='C03'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=None,  name='S02',data_format = "channels_first"))
    model.add(Conv2D(64,3,data_format = "channels_first",
                                                      padding='valid',
                                                      kernel_initializer='random_normal',
                                                      bias_initializer='zeros',
                                                      kernel_regularizer=regularizers.l2(1e-5),
                                                      bias_regularizer = regularizers.l2(1e-5),   name='C03'))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))
    model.add(Conv2D(64,3,data_format = "channels_first",
                                                      padding='valid',
                                                      kernel_initializer='random_normal',
                                                      bias_initializer='zeros',
                                                      kernel_regularizer=regularizers.l2(1e-5),
                                                      bias_regularizer = regularizers.l2(1e-5),   name='C03'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=None,  name='S02',data_format = "channels_first"))
    model.add(Conv2D(128,3,data_format = "channels_first",
                                                      padding='valid',
                                                      kernel_initializer='random_normal',
                                                      bias_initializer='zeros',
                                                      kernel_regularizer=regularizers.l2(1e-5),
                                                      bias_regularizer = regularizers.l2(1e-5),   name='C03'))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))
    model.add(Conv2D(128,3,data_format = "channels_first",
                                                      padding='valid',
                                                      kernel_initializer='random_normal',
                                                      bias_initializer='zeros',
                                                      kernel_regularizer=regularizers.l2(1e-5),
                                                      bias_regularizer = regularizers.l2(1e-5),   name='C03'))
    model.add(Activation('relu'))
    model.add(UpSampling2D(size=(2, 2), data_format="channels_first"))
    model.add(Conv2D(64,3,data_format = "channels_first",
                                                      padding='valid',
                                                      kernel_initializer='random_normal',
                                                      bias_initializer='zeros',
                                                      kernel_regularizer=regularizers.l2(1e-5),
                                                      bias_regularizer = regularizers.l2(1e-5),   name='C03'))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))
    model.add(Conv2D(64,3,data_format = "channels_first",
                                                      padding='valid',
                                                      kernel_initializer='random_normal',
                                                      bias_initializer='zeros',
                                                      kernel_regularizer=regularizers.l2(1e-5),
                                                      bias_regularizer = regularizers.l2(1e-5),   name='C03'))
    model.add(Activation('relu'))
    model.add(UpSampling2D(size=(2, 2), data_format="channels_first"))
    model.add(Conv2D(32,3,data_format = "channels_first",
                                                      padding='valid',
                                                      kernel_initializer='random_normal',
                                                      bias_initializer='zeros',
                                                      kernel_regularizer=regularizers.l2(1e-5),
                                                      bias_regularizer = regularizers.l2(1e-5),   name='C03'))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))
    model.add(Conv2D(32,3,data_format = "channels_first",
                                                      padding='valid',
                                                      kernel_initializer='random_normal',
                                                      bias_initializer='zeros',
                                                      kernel_regularizer=regularizers.l2(1e-5),
                                                      bias_regularizer = regularizers.l2(1e-5),   name='C03'))
    model.add(Activation('relu'))
    
    model.add(Conv2D(1,1,data_format = "channels_first",
                                                      padding='valid',
                                                      kernel_initializer='random_normal',
                                                      bias_initializer='zeros',
                                                      kernel_regularizer=regularizers.l2(1e-5),
                                                      bias_regularizer = regularizers.l2(1e-5),   name='C03'))
    model.add(Activation('relu'))                     
    
    return model
def get_ClassicNet2(n_ch,patch_size):
    model = Sequential()
    model.add(Conv2D(16,4,data_format = "channels_first",activation=None,
                                                      padding='same',
                                                      input_shape=(1,patch_size,patch_size),
                                                      kernel_initializer='random_normal',
                                                      bias_initializer='zeros',
                                                      kernel_regularizer=regularizers.l2(1e-5),
                                                      bias_regularizer = regularizers.l2(1e-5),name='C01'))     
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=None,  name='MP01',data_format = "channels_first"))
    model.add(Conv2D(32,4,data_format = "channels_first",activation=None,
                                                      padding='same',
                                                      kernel_initializer='random_normal',
                                                      bias_initializer='zeros',
                                                      kernel_regularizer=regularizers.l2(1e-5),
                                                      bias_regularizer = regularizers.l2(1e-5),name='C02'))

    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=None,  name='MP02',data_format = "channels_first"))
    model.add(Conv2D(64,4,data_format = "channels_first",activation=None,
                                                      padding='same',
                                                      kernel_initializer='random_normal',
                                                      bias_initializer='zeros',
                                                      kernel_regularizer=regularizers.l2(1e-5),
                                                      bias_regularizer = regularizers.l2(1e-5),name='C03'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=None,  name='MP04',data_format = "channels_first"))
    model.add(Flatten())
    model.add(Dense(128, kernel_initializer='random_normal', 
                                                bias_initializer='zeros',
                                                kernel_regularizer=regularizers.l2(1e-5),
                                                bias_regularizer = regularizers.l2(1e-5)))
    model.add(Activation('relu'))
    model.add(Dense(64, kernel_initializer='random_normal', 
                                                bias_initializer='zeros',
                                                kernel_regularizer=regularizers.l2(1e-5),
                                                bias_regularizer = regularizers.l2(1e-5)))
    model.add(Activation('relu'))
    model.add(Dense(1, kernel_initializer='random_normal', 
                                                bias_initializer='zeros',
                                                kernel_regularizer=regularizers.l2(1e-5)))
    model.compile(loss='binary_crossentropy',
    optimizer=Adam(lr=1e-3, beta_1=0.9, beta_2=0.999, epsilon=1e-5, decay=0),
    metrics=['accuracy'])      
    return model