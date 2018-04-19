# -*- coding: utf-8 -*-
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
import sys
import os
sys.path.insert(0, r'C:\Users\Samsung\Desktop\Retinal Blod Vessel Segementation\Functions')
from PIL import Image
import numpy as np
import fuctionsrep as mf
import extract_patches as ex 
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from matplotlib import pyplot as plt
import random
import tensorflow as tf
sess=tf.Session(config=tf.ConfigProto(log_device_placement=True))
from keras import backend as K
K.tensorflow_backend._get_available_gpus()
# TRAIN MODEL
print('U-Net Implementation')
print('...')
# parameters controlling the training procedure
n_patches_per_img = 100
patch_size = 48
class_ratio = 0.5
n_epochs = 2
ratio_val = 0.25
batch_size = 32
# Training patches
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
patch_img,patch_gt=mf.get_data_training(All_img,All_gt,patch_size,patch_size,n_patches_per_img*n_imgs,True )
patch_gt_2=mf.masks_Unet(patch_gt)
model=mf.get_unet(1,patch_size,patch_size)    
model.summary()
print('Training Model')
print('...')
early_stopping = EarlyStopping(monitor='val_loss', patience=100)
history = model.fit(patch_img,patch_gt_2, 
          batch_size, n_epochs,
          validation_split = ratio_val,
          shuffle = True,
          callbacks=[early_stopping],
          verbose=1)
mf.ModelStats(history)

# Save Model
model_json = model.to_json()
with open("modelU-Net.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("modelU-Net.h5")
print("Saved model to disk")

print('Testing model')
print('...')

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
patch_test,val_01,val_02,pacth_mask=mf.get_data_testing_overlap(All_img_test, All_gt_test,20,patch_size, patch_size,5,5)
predictions = model.predict(patch_test, batch_size=32, verbose=1)
# Reconstructing predititcions to output images
pacth_out=mf.pred_to_imgs(predictions,patch_size,patch_size,'original')
img_out=mf.recompone_overlap(pacth_out, val_01,val_02,5,5)
mf.kill_border(img_out, All_mask_test)
img_out=img_out[:,:,0:img_heigth1,0:img_width]
mf.visualize(mf.group_images(All_img_test,4),"all_originals")
mf.visualize(mf.group_images(img_out,4),"all_predictions")
mf.visualize(mf.group_images(All_gt_test,4),"all_gt")
# Model Metrics and results
y_scores, y_true = mf.pred_only_FOV(img_out,All_gt_test, All_mask_test)
fpr, tpr, thresholds = roc_curve((y_true), y_scores)
mf.get_metrics(y_true, y_scores, 0.5, fpr, tpr, thresholds)

