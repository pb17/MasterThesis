# -*- coding: utf-8 -*-
"""
Created on Wed May 30 17:14:52 2018

@author: Samsung
"""
from keras import regularizers
from keras.optimizers import Adam, RMSprop
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Activation, Input
from keras.layers import Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.utils import np_utils
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.models import model_from_json, load_model
from sklearn.ensemble import RandomForestClassifier
import sys
import os
from keras.models import Model
sys.path.insert(0, r'C:\Users\Samsung\Desktop\Retinal Blod Vessel Segementation\Functions')
#sys.path.insert(0, r'../functions')
#os.environ["CUDA_VISIBLE_DEVICES"]=""
from PIL import Image
import numpy as np
import fuctionsrep as mf
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from matplotlib import pyplot as plt
from sklearn import preprocessing
from sklearn.ensemble import VotingClassifier
import time
import tensorflow as tf
from sklearn.feature_selection import VarianceThreshold
sess=tf.Session(config=tf.ConfigProto(log_device_placement=True))
from keras import backend as K
K.tensorflow_backend._get_available_gpus()
# TRAIN MODEL
print('...')
print('CNN as Feature extractor - Random Forests')
print('...')
# parameters controlling the training procedure
n_patches_per_img = 10500
patch_size = 25
class_ratio = 0.35
n_epochs = 150
ratio_val = 0.25
batch_size = 32
model = load_model('Complet_Model')

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
X_train_n = X_train
X_train_n = X_train_n.reshape(X_train_n.shape[0], 1, patch_size, patch_size)
X_train_final=X_train_n[0:int(n_patches_per_img*(1-ratio_val)*n_imgs),:,:,:]
Y_train_final=Y_train[0:int(n_patches_per_img*(1-ratio_val)*n_imgs)]
X_val=X_train_n[int(n_patches_per_img*(1-ratio_val)*n_imgs):,:,:,:]
Y_val=Y_train[int(n_patches_per_img*(1-ratio_val)*n_imgs):]
Val_set=(X_val,Y_val)
# Feature Maps
# Feature Map01
print('Get Feature map01')
X_batch=X_train_n
out = model.get_layer(index=3).output
layerout_new = Model(model.input, out)
C1 = layerout_new.predict(X_batch)
C1 = np.squeeze(C1)
# Feature Map02
print('Get Feature map02')
out = model.get_layer(index=6).output
layerout_new = Model(model.input, out)
C2 = layerout_new.predict(X_batch)
C2 = np.squeeze(C2)
# Feature Map03
print('Get Feature map03')
out = model.get_layer(index=9).output
layerout_new = Model(model.input, out)
C3 = layerout_new.predict(X_batch)
C3 = np.squeeze(C3)
inclass01=np.zeros([C1.shape[0],C1[0].size])
inclass02=np.zeros([C2.shape[0],C2[0].size])
for i in range(X_train.shape[0]): 
 inclass01[i]=mf.get_cenas(C1[i,:,:,:])
for i in range(X_train.shape[0]): 
 inclass02[i]=mf.get_cenas(C2[i,:,:,:])
 
# Classifiers 
print('Training Class01')
clf01 = RandomForestClassifier(n_estimators=10,criterion='gini')
clf01.fit(inclass01, Y_train)
print('Training Class02') 
clf02 = RandomForestClassifier(n_estimators=10,criterion='gini')
clf02.fit(inclass02, Y_train)
print('Training Class03')
clf03 = RandomForestClassifier(n_estimators=10,criterion='gini')
clf03.fit(C3, Y_train)

n_imgs_test=20
for i in range(n_imgs_test):    
    i=0
    print("Evaluation on test image ",i, "...")
    # Get Images
    img_path = X_test_folder + os.listdir(X_test_folder)[i]
    gt_path  = Y_test_folder + os.listdir(Y_test_folder)[i]
    val_path = v_test_folder + os.listdir(v_test_folder)[i]
    # Get Data and Patches
    img, gt, val_mask = mf.getImageData (img_path, gt_path, val_path)
    X_test, Y_test,my_pos_set, my_neg_set  = mf.getTestPatches(img, gt, val_mask, patch_size)
    X_test = X_test.reshape(X_test.shape[0], 1, patch_size, patch_size)
    X_test = X_test.astype('float32')    
    # Feature extractor
    X_batch=X_test
    print('Feature Test 01')
    out = model.get_layer(index=3).output
    layerout_new = Model(model.input, out)
    C1 = layerout_new.predict(X_batch)
    C1 = np.squeeze(C1)
    out = model.get_layer(index=6).output
    layerout_new = Model(model.input, out)
    C2 = layerout_new.predict(X_batch)
    C2 = np.squeeze(C2)
    print('Feature Test 02')
    out = model.get_layer(index=9).output
    layerout_new = Model(model.input, out)
    C3 = layerout_new.predict(X_batch)
    C3 = np.squeeze(C3)
    print('Feature Test 03')
    inclass01=np.zeros([C1.shape[0],C1[0].size])
    inclass02=np.zeros([C2.shape[0],C2[0].size])
    predictions01=np.zeros([X_test.shape[0]])
    predictions02=np.zeros([X_test.shape[0]])
    predictions03=np.zeros([X_test.shape[0]])
    predFinal=np.zeros([X_test.shape[0]])
    #Reduce To 2D 
    for l in range(X_test.shape[0]): 
     inclass01[l]=mf.get_cenas(C1[i,:,:,:])   
    for l in range(X_test.shape[0]): 
     inclass02[l]=mf.get_cenas(C2[i,:,:,:])
    print('Make prediction') 
    for k in range(X_test.shape[0]):
     predictions01[k]=clf01.predict(inclass01[i].reshape(1, -1))
     predictions02[k]=clf02.predict(inclass02[i].reshape(1, -1))
     predictions03[k]=clf03.predict(C3[i].reshape(1, -1))
    #Metrics Pred Raw
    np.save('PredictionRF_Vector01_%s'%i,  predictions01)
    np.save('PredictionRF_Vector02_%s'%i,  predictions02)
    np.save('PredictionRF_Vector03_%s'%i,  predictions03)
    print('\n\n RESULTS RF01')
    fpr, tpr, thresholds = roc_curve(Y_test,predictions01)
    mf.get_metrics(Y_test,predictions01, 0.5, fpr, tpr, thresholds)
    print('\n\n RESULTS RF02')
    fpr, tpr, thresholds = roc_curve(Y_test,predictions02)
    mf.get_metrics(Y_test,predictions02, 0.5, fpr, tpr, thresholds)
    print('\n\n RESULTS RF03')
    fpr, tpr, thresholds = roc_curve(Y_test,predictions03)
    mf.get_metrics(Y_test,predictions03, 0.5, fpr, tpr, thresholds)
    # Ensemble:
    # Max
    print('\n\n Emsemble Maximo')
    predFinal=np.max(np.column_stack((predictions01,predictions02,predictions03)), axis=1)
    fpr, tpr, thresholds = roc_curve(Y_test,predFinal)
    predFinal[predFinal >=0.5]=1
    predFinal[predFinal <0.5 ]=0
    mf.get_metrics(Y_test,predFinal, 0.5, fpr, tpr, thresholds)
    # Mean
    print('\n\n Emsemble Media')
    predFinal=np.mean(np.column_stack((predictions01,predictions02,predictions03)), axis=1)
    fpr, tpr, thresholds = roc_curve(Y_test,predFinal)
    mf.get_metrics(Y_test,predFinal, 0.5, fpr, tpr, thresholds)
    # Median
    print('\n\n Emsemble Median')
    predFinal=np.median(np.column_stack((predictions01,predictions02,predictions03)), axis=1)
    fpr, tpr, thresholds = roc_curve(Y_test,predFinal)
    mf.get_metrics(Y_test,predFinal, 0.5, fpr, tpr, thresholds)
    #Fuzzy like Sistem
    print('\n\n Emsemble Fuzzy')
    wpred01=np.count_nonzero(predictions01==Y_test)/Y_test.shape[0]
    wpred02=np.count_nonzero(predictions02==Y_test)/Y_test.shape[0]
    wpred03=np.count_nonzero(predictions03==Y_test)/Y_test.shape[0]
    wVector=np.corrcoef(np.asarray([wpred01,wpred02,wpred03]))
    for j in range (X_test.shape[0]): 
        predFinal[j]=(predictions01[j]*((wpred01*0.65/0.9))+predictions02[j]*((wpred01*0.65/0.9))+predictions03[j]*((wpred01*0.65/0.9)))/(1+predictions02[j]+predictions03[j])
    fpr, tpr, thresholds = roc_curve(Y_test,predFinal)
    predFinal[predFinal >=0.5]=1
    predFinal[predFinal <0.5 ]=0
    mf.get_metrics(Y_test,predFinal, 0.5, fpr, tpr, thresholds)

    






