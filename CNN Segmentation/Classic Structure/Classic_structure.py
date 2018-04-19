# -*- coding: utf-8 -*-
"""
Classic CNN Implementation 
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
print('Liskowski - Augmented Data Implementation')
print('...')
# parameters controlling the training procedure
n_patches_per_img = 500
patch_size = 25
class_ratio = 0.5
n_epochs = 25
ratio_val = 0.25
batch_size = 32
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