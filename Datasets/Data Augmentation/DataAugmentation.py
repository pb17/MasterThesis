# -*- coding: utf-8 -*-
"""
Created on Thu Apr 12 19:15:01 2018

@author: Samsung
"""
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
from keras.preprocessing.image import ImageDataGenerator
seed = 7
import tensorflow as tf
sess=tf.Session(config=tf.ConfigProto(log_device_placement=True))
from keras import backend as K
K.tensorflow_backend._get_available_gpus()

X_train_folder,Y_train_folder,v_train_folder,X_test_folder,Y_test_folder,v_test_folder, n_imgs,img_width, img_heigth1=mf.GetSTAREPRE()

img_list=[]
gt_list=[]
mask_list=[]
datagen = ImageDataGenerator(
        featurewise_center=True,
        featurewise_std_normalization=True,
        rotation_range=180,
        shear_range=0.2,
        horizontal_flip=True,
        vertical_flip=True)

for i in range(10):
      img_path = X_train_folder + os.listdir(X_train_folder)[i]
      gt_path  = Y_train_folder + os.listdir(Y_train_folder)[i]
      val_path = v_train_folder + os.listdir(v_train_folder)[i]
      img, gt, val_mask = mf.getImageData (img_path, gt_path, val_path)
      #All_img, All_gt, All_mask=mf.getdataok(img_list,gt_list,mask_list)
      img=img.reshape(1,605,700,3)
##img = img.reshape((1,) + img.shape) 
##img = img.reshape((1,) + img.shape) 
      j=0
      for batch in datagen.flow(img, batch_size=10,
                                save_to_dir=r'C:/Users/Samsung/Desktop/Retinal Blod Vessel Segementation/Datasets/Data Augmentation', save_prefix='Aug', save_format='ppm'):
       j += 1
       if j > 10:
        break 