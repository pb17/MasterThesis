# -*- coding: utf-8 -*-
"""
'CNN as Feature extractor - Random Forests'
"""
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
from sklearn.ensemble import RandomForestClassifier
import sys
import os
sys.path.insert(0, r'C:\Users\Samsung\Desktop\Retinal Blod Vessel Segementation\Functions')
from PIL import Image
import numpy as np
import fuctionsrep as mf
import extract_patches as ex
import nets as net 
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from matplotlib import pyplot as plt
import random
import tensorflow as tf
sess=tf.Session(config=tf.ConfigProto(log_device_placement=True))
from keras import backend as K
K.tensorflow_backend._get_available_gpus()
# TRAIN MODEL
print('CNN as Feature extractor - Random Forests')
print('...')
# parameters controlling the training procedure
n_patches_per_img = 10
patch_size = 25
class_ratio = 0.5
n_epochs = 10
ratio_val = 0.25
batch_size = 32
# Training patches
#X_train_folder,Y_train_folder,v_train_folder,X_test_folder,Y_test_folder,v_test_folder, n_imgs,img_width, img_heigth1=mf.GetDRIVEPREAUG()
#X_train_folder,Y_train_folder,v_train_folder,X_test_folder,Y_test_folder,v_test_folder, n_imgs,img_width, img_heigth1=mf.GetSTAREPREAUG()
X_train_folder,Y_train_folder,v_train_folder,X_test_folder,Y_test_folder,v_test_folder, n_imgs,img_width, img_heigth1=mf.GetDRIVEPRE()
X_train  = np.ndarray([n_patches_per_img*n_imgs,patch_size,patch_size])
Y_train = np.ndarray([n_patches_per_img*n_imgs])

for i in range(n_imgs):
    img_path = X_train_folder + os.listdir(X_train_folder)[i]
    gt_path  = Y_train_folder + os.listdir(Y_train_folder)[i]
    val_path = v_train_folder + os.listdir(v_train_folder)[i]    
    img, gt, val_mask = mf.getImageData (img_path, gt_path, val_path)
    X_train[i*n_patches_per_img:(i+1)*n_patches_per_img,:,:], Y_train[i*n_patches_per_img:(i+1)*n_patches_per_img] = mf.getTrainPatches(img, gt, val_mask, n_patches_per_img, patch_size, class_ratio)

X_train = X_train.astype('float32')
X_train_n = np.ndarray(X_train.shape)
for i in range(X_train.shape[0]):
    X_train_n[i,:,:] = (X_train[i,:,:] - np.mean(X_train[i,:,:])) / (np.std(X_train[i,:,:])+0.00001)
#Validatio Split
X_train_n = X_train_n.reshape(X_train_n.shape[0], 1, patch_size, patch_size)
X_train_final=X_train_n[0:int(n_patches_per_img*(1-ratio_val)*n_imgs),:,:,:]
Y_train_final=Y_train[0:int(n_patches_per_img*(1-ratio_val)*n_imgs)]
X_val=X_train_n[int(n_patches_per_img*(1-ratio_val)*n_imgs):,:,:,:]
Y_val=Y_train[int(n_patches_per_img*(1-ratio_val)*n_imgs):]
Val_set=(X_val,Y_val)
# Get Net
model = Sequential()
model.add(Conv2D(12,4,data_format = "channels_first",activation=None,
                                                      padding='same',
                                                      input_shape=(1,patch_size,patch_size),
                                                      kernel_initializer='random_normal',
                                                      bias_initializer='zeros',
                                                      kernel_regularizer=regularizers.l2(1e-5),
                                                      bias_regularizer = regularizers.l2(1e-5),name='C01'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=None,  name='S01',data_format = "channels_first"))
model.add(Dropout(0.5))
model.add(Conv2D(12,4,data_format = "channels_first",
                                                      padding='valid',
                                                      kernel_initializer='random_normal',
                                                      bias_initializer='zeros',
                                                      kernel_regularizer=regularizers.l2(1e-5),
                                                      bias_regularizer = regularizers.l2(1e-5),   name='C03'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=None,  name='S02',data_format = "channels_first"))
model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(100, kernel_initializer='random_normal', 
                                                bias_initializer='zeros',
                                                kernel_regularizer=regularizers.l2(1e-5),
                                                bias_regularizer = regularizers.l2(1e-5),   name='C04'))
model.add(Activation('relu'))
model.add(Dense(1, kernel_initializer='random_normal', 
                                                bias_initializer='zeros',
                                                kernel_regularizer=regularizers.l2(1e-5),
                                                bias_regularizer = regularizers.l2(1e-5),   name='C05'))

model.add(Activation('sigmoid'))

model.compile(loss='binary_crossentropy',
        optimizer=Adam(lr=1e-3, beta_1=0.9, beta_2=0.999, epsilon=1e-5, decay=0),
        metrics=['accuracy']) 
model.summary()
early_stopping = EarlyStopping(monitor='val_loss', patience=100)
history = model.fit(X_train_final,Y_train_final, 
          batch_size, n_epochs,
           validation_data = Val_set,
          shuffle = True,
          callbacks=[early_stopping],
          verbose=1)
mf.ModelStats(history)
# Save
# serialize model to JSON
model_json = model.to_json()
with open("model_Feature_extractor_RF", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model_Feature_extractor_RF.h5")

# Get Features Maps
#for i in range(patch_img.shape[0]):    
inclass01=[]
inclass02=[]
inclass03=[]

for i in range(X_train_final.shape[0]):
    test_image=X_train_final[i]
    test_image=np.expand_dims(test_image, axis=0)
    activations01 = np.asarray(mf.get_featuremaps(model, int(2),test_image))
    activations02 = np.asarray(mf.get_featuremaps(model, int(6),test_image))
    activations03 = np.asarray(mf.get_featuremaps(model, int(9), test_image))
    activations01 = activations01[0,0,:,:,:]
    activations02 = activations02[0,0,:,:,:]
    activations03 =activations03[0,0,:]
    inclass01.append(mf.test(activations01))
    inclass02.append(mf.test(activations02))
    inclass03.append(activations03)

inclass01=np.asarray(inclass01)
inclass02=np.asarray(inclass02)
inclass03=np.asarray(inclass03)
#Classifer 01
clf01 = RandomForestClassifier(n_estimators=100,criterion='gini', max_features=4,class_weight='balanced' )
clf01.fit(inclass01, Y_train_final)
# Classifer 02
clf02 = RandomForestClassifier(n_estimators=100,criterion='gini', max_features=4,class_weight='balanced' )
clf02.fit(inclass02, Y_train_final)
# Classifer 03
clf03 = RandomForestClassifier(n_estimators=100,criterion='gini', max_features=4,class_weight='balanced' )
clf03.fit(inclass03, Y_train_final)
#emsemble methodos 
n_imgs_test=1
for i in range(n_imgs_test):    
    print("Evaluation on test image ",i, "...")
    # Get Images
    img_path = X_test_folder + os.listdir(X_test_folder)[i]
    gt_path  = Y_test_folder + os.listdir(Y_test_folder)[i]
    val_path = v_test_folder + os.listdir(v_test_folder)[i]
    # Get Data and Patches
    img, gt, val_mask = mf.getImageData (img_path, gt_path, val_path)
    X_test, Y_test,N_patches, Final_out, Dim01, Dim02  = mf.getTestPatches(img, gt, val_mask, patch_size)
    X_test = X_test.reshape(X_test.shape[0], 1, patch_size, patch_size)
    X_test = X_test.astype('float32')
    for i in range(X_test.shape[0])
      predictions01=clf01.predict(X_test[0,0,:,:])
      predictions02=clf02.predict(X_test)
      predictions03=clf03.predict(X_test)
