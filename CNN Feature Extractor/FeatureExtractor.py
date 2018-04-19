# -*- coding: utf-8 -*-
"""
@author: Samsung
"""
# Keras
from keras import regularizers
from keras.optimizers import Adam, SGD, Adagrad
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Activation, Input
from keras.layers import Conv2D, MaxPooling2D,Reshape
from keras.layers.normalization import BatchNormalization
from keras.utils import np_utils
from keras.callbacks import EarlyStopping
from keras.utils import plot_model
#scikit learn
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import jaccard_similarity_score
from sklearn.metrics import f1_score
from sklearn.ensemble import RandomForestClassifier
import os
# Skimage
from scipy.ndimage.interpolation import rotate
from skimage import io

from PIL import Image
import numpy as np
from matplotlib import pyplot as plt
import myfunctions as mf
import extract_patches as ex
import pandas as pd
seed = 7
import tensorflow as tf
sess=tf.Session(config=tf.ConfigProto(log_device_placement=True))
config = tf.ConfigProto(allow_soft_placement=True)
config.gpu_options.allow_growth = True
print('Runnig on GPU')
print('training model')
print('...')
n_patches_per_img = 1
patch_size = 24
class_ratio = 0.5
n_epochs = 1
ratio_val = 0.1
n_imgs = 20
batch_size = 32

X_train_folder,Y_train_folder,v_train_folder,X_test_folder,Y_test_folder,v_test_folder, n_imgs,img_width, img_heigth1=mf.GetDRIVEPRE()
img_list=[]
gt_list=[]
mask_list=[]
for i in range(n_imgs):
    img_path = X_train_folder + os.listdir(X_train_folder)[i]
    gt_path  = Y_train_folder + os.listdir(Y_train_folder)[i]
    val_path = v_train_folder + os.listdir(v_train_folder)[i]
    img, gt, val_mask = mf.getImageData (img_path, gt_path, val_path)
    img_list.append(img)
    gt_list.append(gt)
    mask_list.append(val_mask)
    
All_img, All_gt, All_mask=mf.getdataok(img_list,gt_list,mask_list)
patch_img,patch_gt=ex.get_data_training(All_img,All_gt,patch_size,patch_size,n_patches_per_img*n_imgs, False)
gt_vector=[]
for i in range(patch_gt.shape[0]):
    if 1 in patch_gt[i]:
        gt_vector.append(1)
    else:
       if 0 in patch_gt[i]:
        gt_vector.append(0)
gt_vector=np.asarray(gt_vector)        
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
hist = model.fit(patch_img,gt_vector, 
          batch_size, n_epochs,
          validation_split = ratio_val,
          shuffle = True,
          callbacks=[early_stopping],
          verbose=1)
model_json = model.to_json()
with open("modelFeatureExtractor.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("modelFeatureExtractor.h5")
print("Saved model to disk")
# Get Features Maps
#for i in range(patch_img.shape[0]):    
inclass01=[]
inclass02=[]
inclass03=[]

for i in range(patch_img.shape[0]):
    test_image=patch_img[i]
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
# Get Test
img_list_test=[]
gt_list_test=[]
mask_list_test=[]

for i in range(n_imgs):
    img_path = X_test_folder + os.listdir(X_test_folder)[i]
    gt_path  = Y_test_folder + os.listdir(Y_test_folder)[i]
    val_path = v_test_folder + os.listdir(v_test_folder)[i]
    img, gt, val_mask = mf.getImageData (img_path, gt_path, val_path)
    img_list_test.append(img)
    gt_list_test.append(gt)
    mask_list_test.append(val_mask)

All_img_test, All_gt_test, All_mask_test=mf.getdataok(img_list_test,gt_list_test,mask_list_test)
patch_imgs_test,B,C,patch_gt_test=ex.get_data_testing_overlap(All_img_test,All_gt_test,20,patch_size, patch_size,5,5)
# Classificador 01
clf01 = RandomForestClassifier(n_estimators=100,criterion='gini', max_features='auto',class_weight='balanced' )
clf01.fit(inclass01, gt_vector)
# Classificador 02
clf02 = RandomForestClassifier(n_estimators=100,criterion='gini', max_features='auto',class_weight='balanced' )
clf02.fit(inclass02, gt_vector)
#clf02.socre(x, gt_vector)
# Classificador 03
clf03 = RandomForestClassifier(n_estimators=100,criterion='gini', max_features='auto',class_weight='balanced' )
clf03.fit(inclass03, gt_vector)
#clf03.socre(x, gt_vector) 
# Emsemble
predictions01=clf01.predict(patch_imgs_test)
predictions02=clf02.predict(patch_imgs_test)
predictions03=clf03.predict(patch_imgs_test)
