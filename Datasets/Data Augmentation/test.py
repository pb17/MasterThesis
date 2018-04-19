# -*- coding: utf-8 -*-
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
import os
# Skimage
from scipy.ndimage.interpolation import rotate
from skimage import io
from PIL import Image
import numpy as np
from matplotlib import pyplot as plt
import myfunctions as mf
import extract_patches as ex
import nets
import pandas as pd
import random
seed = 7
import tensorflow as tf
sess=tf.Session(config=tf.ConfigProto(log_device_placement=True))
from keras import backend as K
K.tensorflow_backend._get_available_gpus()

X_train_folder,Y_train_folder,v_train_folder,X_test_folder,Y_test_folder,v_test_folder, n_imgs,img_width, img_heigth1=mf.GetDRIVEPRE()

n_patches_per_img = 1000
patch_size = 21
class_ratio = 0.5
n_epochs = 1
ratio_val = 0.2
n_imgs = 20
batch_size = 64



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

X_train_n = X_train_n.reshape(X_train_n.shape[0], 1, patch_size, patch_size)
X_train_Out = mf.ZCAW(X_train_n, patch_size)