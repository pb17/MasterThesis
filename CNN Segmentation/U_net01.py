# -*- coding: utf-8 -*-
"""
Created on Thu Mar 29 18:54:09 2018

@author: Samsung
"""
#Retinal Vessel Segmentatition 
#Implementatiom pacth a pacth
#Keras
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
#scikit learn
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import jaccard_similarity_score
from sklearn.metrics import f1_score
from scipy.ndimage.interpolation import rotate
import os
from PIL import Image
import numpy as np
from matplotlib import pyplot as plt
import random
#System
import sys
import myfunctions
import extract_patches as ex 
import pandas as pd
seed = 7
import random
import tensorflow as tf
sess=tf.Session(config=tf.ConfigProto(log_device_placement=True))
from keras import backend as K
K.tensorflow_backend._get_available_gpus()
# TRAIN MODEL
print('training model')
print('...')
# parameters controlling the training procedure
n_patches_per_img = 10
patch_size = 48
class_ratio = 0.5
n_epochs = 1
ratio_val = 0.1
n_imgs = 20
batch_size = 32

X_train_folder,Y_train_folder,v_train_folder,X_test_folder,Y_test_folder,v_test_folder, n_imgs,img_width, img_heigth1=myfunctions.GetDRIVEPRE()

img_list=[]
gt_list=[]
mask_list=[]
for i in range(n_imgs):
    img_path = X_train_folder + os.listdir(X_train_folder)[i]
    gt_path  = Y_train_folder + os.listdir(Y_train_folder)[i]
    val_path = v_train_folder + os.listdir(v_train_folder)[i]
    img, gt, val_mask = myfunctions.getImageData (img_path, gt_path, val_path)
    img_list.append(img)
    gt_list.append(gt)
    mask_list.append(val_mask)
    
All_img, All_gt, All_mask=myfunctions.getdataok(img_list,gt_list,mask_list)
patch_img,patch_gt=ex.get_data_training(All_img,All_gt,patch_size,patch_size,n_patches_per_img*n_imgs,True )
patch_gt_2=myfunctions.masks_Unet(patch_gt)    
model=myfunctions.get_unet(1,patch_size,patch_size)    
early_stopping = EarlyStopping(monitor='val_loss', patience=100)
history = model.fit(patch_img,patch_gt_2, 
          batch_size, n_epochs,
          validation_split = ratio_val,
          shuffle = True,
          callbacks=[early_stopping],
          verbose=1)
myfunctions.ModelStats(history)
print('Testing Model')
print('....')

img_list_test=[]
gt_list_test=[]
mask_list_test=[]

for i in range(n_imgs):
    img_path = X_test_folder + os.listdir(X_test_folder)[i]
    gt_path  = Y_test_folder + os.listdir(Y_test_folder)[i]
    val_path = v_test_folder + os.listdir(v_test_folder)[i]
    img, gt, val_mask = myfunctions.getImageData (img_path, gt_path, val_path)
    img_list_test.append(img)
    gt_list_test.append(gt)
    mask_list_test.append(val_mask)

All_img_test, All_gt_test, All_mask_test=myfunctions.getdataok(img_list_test,gt_list_test,mask_list_test)
qualquercoisa,B,C,patches_imgs_test=ex.get_data_testing_overlap(All_img_test,All_gt_test,20,patch_size, patch_size,5,5)
predictions = model.predict(qualquercoisa, batch_size=32, verbose=2)
pacths_out=myfunctions.pred_to_imgs(predictions,patch_size,patch_size,'original')
imgout=ex.recompone_overlap(pacths_out, B,C,5,5)
ex.kill_border(imgout, All_mask_test)
imgout=imgout[:,:,0:img_heigth1,0:img_width]
myfunctions.visualize(myfunctions.group_images(All_img_test,4),"all_originals_2")#.show()
myfunctions.visualize(myfunctions.group_images(imgout,4),"all_predictions_2")#.show()
myfunctions.visualize(myfunctions.group_images(All_gt_test,4),"all_gt_2")
y_scores, y_true = myfunctions.pred_only_FOV(imgout,All_gt_test, All_mask_test)



