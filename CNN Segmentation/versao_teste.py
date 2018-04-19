# -*- coding: utf-8 -*-
"""
Deep Features extractor
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
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import jaccard_similarity_score
from sklearn.metrics import f1_score
from sklearn.ensemble import RandomForestClassifier
import os
# Skimage
from scipy.ndimage.interpolation import rotate
from skimage import io
from sklearn.feature_extraction import image
from PIL import Image
import numpy as np
from matplotlib import pyplot as plt
import myfunctions as mf
import extract_patches as ex
import random
import xlsxwriter
import pandas as pd
seed = 7
import tensorflow as tf
sess=tf.Session(config=tf.ConfigProto(log_device_placement=True))
config = tf.ConfigProto(allow_soft_placement=True)
config.gpu_options.allow_growth = True

n_patches_per_img = 30000
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
# Show Superxil example
#mf.showsuperpixel(All_img)
img_patchs,gt_patchs=mf.get_data_training_deepFeatures(All_img,All_gt,20,24,24,2, 2)



test=mf.paint_border_overlap(All_gt, 24, 24, 2, 2)



