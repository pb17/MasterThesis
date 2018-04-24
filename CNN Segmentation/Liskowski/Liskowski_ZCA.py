# -*- coding: utf-8 -*-
"""
Liskowski implementation - Data Augmentation
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
def zca_whitening_matrix(X):
       sigma = np.cov(X, rowvar=True) 
       U,S,V = np.linalg.svd(sigma)
       epsilon = 1e-5
       ZCAMatrix = np.dot(U, np.dot(np.diag(1.0/np.sqrt(S + epsilon)), U.T))
       
       return ZCAMatrix

# TRAIN MODEL
print('Liskowski - ZCA Implementation')
print('...')
# parameters controlling the training procedure
n_patches_per_img = 1000
patch_size = 27
class_ratio = 0.5
n_epochs = 150
ratio_val = 0.25
batch_size = 100
# Training patches
#X_train_folder,Y_train_folder,v_train_folder,X_test_folder,Y_test_folder,v_test_folder, n_imgs,img_width, img_heigth1=mf.GetSTAREPRE()
X_train_folder,Y_train_folder,v_train_folder,X_test_folder,Y_test_folder,v_test_folder, n_imgs,img_width, img_heigth1=mf.GetDRIVEPRE()
X_train  = np.ndarray([n_patches_per_img*n_imgs,patch_size,patch_size])
Y_train = np.ndarray([n_patches_per_img*n_imgs])
for i in range(n_imgs):

    img_path = X_train_folder + os.listdir(X_train_folder)[i]
    gt_path  = Y_train_folder + os.listdir(Y_train_folder)[i]
    val_path = v_train_folder + os.listdir(v_train_folder)[i]    
    img, gt, val_mask = mf.getImageData (img_path, gt_path, val_path)
    #img, gt, val_mask = mf.getImageDataGreen (img_path, gt_path, val_path)
    X_train[i*n_patches_per_img:(i+1)*n_patches_per_img,:,:], Y_train[i*n_patches_per_img:(i+1)*n_patches_per_img] = mf.getTrainPatches(img, gt, val_mask, n_patches_per_img, patch_size, class_ratio)

X_train_n = X_train.astype('float32')
X_train_n = np.ndarray(X_train.shape)
for i in range(X_train.shape[0]):
    X_train_n[i,:,:] = (X_train[i,:,:] - np.mean(X_train[i,:,:])) / (np.std(X_train[i,:,:])+0.00001)

X_train_n = X_train_n.reshape(X_train_n.shape[0], 1, patch_size, patch_size)
OutZCAMatrix=[]
for i in range(X_train_n.shape[0]): 
        X=X_train_n[i,0]
        ZCAMatrix = zca_whitening_matrix(X) # get ZCAMatrix
        ZCAMatrix # [5 x 5] matrix
        xZCAMatrix = np.dot(ZCAMatrix, X) # project X onto the ZCAMatrix
        OutZCAMatrix.append(xZCAMatrix)    
OutZCAMatrix=np.asarray(OutZCAMatrix)
OutZCAMatrix= OutZCAMatrix.reshape(X_train_n.shape[0], 1, patch_size, patch_size)   
X_train_n=OutZCAMatrix
X_train_final=X_train_n[0:int(n_patches_per_img*(1-ratio_val)*n_imgs),:,:,:]
Y_train_final=Y_train[0:int(n_patches_per_img*(1-ratio_val)*n_imgs)]
X_val=X_train_n[int(n_patches_per_img*(1-ratio_val)*n_imgs):,:,:,:]
Y_val=Y_train[int(n_patches_per_img*(1-ratio_val)*n_imgs):]
Val_set=(X_val,Y_val)
model=net.get_Liskowski(patch_size,patch_size)
model.summary()
early_stopping = EarlyStopping(monitor='val_loss', patience=100)
history = model.fit(X_train_n, Y_train, 
          batch_size, n_epochs,
          validation_split = ratio_val,
          shuffle = True,
          callbacks=[early_stopping],
          verbose=1)
mf.ModelStats(history)
# Save
# serialize model to JSON
model_json = model.to_json()
with open("model_Liskowski_ZCA.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model_Liskowski_ZCA.h5")
print("Saved model to disk")
print('Testing model')
print('...')
X_test = []
Y_test = []
n_imgs_test=10
for i in range(1):    
    print("Evaluation on test image ",i, "...")
    # Get Images
    img_path = X_test_folder + os.listdir(X_test_folder)[i]
    gt_path  = Y_test_folder + os.listdir(Y_test_folder)[i]
    val_path = v_test_folder + os.listdir(v_test_folder)[i]
    # Get Data and Patches
    img, gt, val_mask = mf.getImageData (img_path, gt_path, val_path)
    X_test, Y_test,my_pos_set,my_neg_set = mf.getTestPatches(img, gt, val_mask, patch_size)
#    X_test, Y_test,N_patches, Final_out, Dim01, Dim02  = mf.getTestPatches(img, gt, val_mask, patch_size)
    X_test = X_test.reshape(X_test.shape[0], 1, patch_size, patch_size)
    X_test = X_test.astype('float32')
    prediction=model.predict(X_test,batch_size=32, verbose=1)
    scores = model.evaluate(X_test, Y_test, verbose=1)
#    pred_patches=mf.conv_predriction(prediction,N_patches, patch_size, rec_mode='Original')
#    img_out=mf.image_reconstruct(pred_patches, patch_size, N_patches, Dim01,Dim02)
#    img_out=img_out[:,:,0:img_heigth1,0:img_width]
#    img_out_viz=img_out[0,0,:,:]
#    img_out_viz=np.reshape(img_out_viz,(img_out_viz.shape[0],img_out_viz.shape[1],1))
#    mf.visualize(img_out_viz,"predicitons_%s.png" % i)
#    fpr, tpr, thresholds = roc_curve(Y_test,prediction)
#    mf.get_metrics(Y_test,prediction, 0.5, fpr, tpr, thresholds)
    out_img=mf.Image_reconstruct(my_pos_set, my_neg_set, prediction, 'DRIVE','prob')
    img_out_viz=np.reshape(out_img,(out_img.shape[0],out_img.shape[1],1))
    mf.visualize(img_out_viz,"predicitons_%s.png" % i)
    pred_clas=np.empty(prediction.shape[0])
    pred_clas[prediction >0.5]=1
    pred_clas[prediction <0.49]=0
    fpr, tpr, thresholds = roc_curve(Y_test,pred_clas)
    mf.get_metrics(Y_test,pred_clas, 0.5, fpr, tpr, thresholds)