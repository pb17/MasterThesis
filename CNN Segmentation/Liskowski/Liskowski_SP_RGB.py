# -*- coding: utf-8 -*-

"""
Liskowski implementation - Structure Prediction 
"""
from keras import regularizers
from keras.optimizers import Adam, SGD, Adagrad,RMSprop
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Activation, Input
from keras.layers import Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.utils import np_utils
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.utils import plot_model
from keras.models import model_from_json
import sys
import os
sys.path.insert(0, r'C:\Users\Samsung\Desktop\Retinal Blod Vessel Segementation\Functions')
#sys.path.insert(0, r'../functions')
#os.environ["CUDA_VISIBLE_DEVICES"]=""
from PIL import Image
import numpy as np
import fuctionsrep as mf
import nets as net 
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from matplotlib import pyplot as plt
from sklearn import preprocessing
import random
import tensorflow as tf
sess=tf.Session(config=tf.ConfigProto(log_device_placement=True))
from keras import backend as K
K.tensorflow_backend._get_available_gpus()
# TRAIN MODEL
print('...')
print('Liskowski - Structure Prediction')
print('...')
# parameters controlling the training procedure
n_patches_per_img = 10500
patch_size = 17
class_ratio = 0.50
n_epochs = 100
ratio_val = 0.25
batch_size = 32
# Training patches

X_train_folder,Y_train_folder,v_train_folder,X_test_folder,Y_test_folder,v_test_folder, n_imgs,img_width, img_heigth1=mf.GetDRIVE()
X_train  = np.ndarray([n_patches_per_img*n_imgs,patch_size,patch_size,3])
Y_train = np.ndarray([n_patches_per_img*n_imgs])
SP=np.ndarray([9, Y_train.shape[0]])
for i in range(n_imgs):

    img_path = X_train_folder + os.listdir(X_train_folder)[i]
    gt_path  = Y_train_folder + os.listdir(Y_train_folder)[i]
    val_path = v_train_folder + os.listdir(v_train_folder)[i]    
    img, gt, val_mask = mf.getImageData (img_path, gt_path, val_path)
    X_train[i*n_patches_per_img:(i+1)*n_patches_per_img,:,:,:], Y_train[i*n_patches_per_img:(i+1)*n_patches_per_img], SP[:,i*n_patches_per_img:(i+1)*n_patches_per_img] = mf.getTrainPatchesLisk(img, gt, val_mask, n_patches_per_img, patch_size, class_ratio)

X_train = X_train.astype('float32')
#X_train_n =np.empty(X_train.shape)
#for i in range(X_train.shape[0]):
##   X_train_n[i,:,:] = preprocessing.scale(X_train[i,:,:])
#    X_train_n[i,:,:] =X_train[i]/np.max(np.abs(X_train[i]))
X_train_n = X_train.reshape(X_train.shape[0], 3, patch_size, patch_size)
X_train_final=X_train_n[0:int(n_patches_per_img*(1-ratio_val)*n_imgs),:,:,:]
Y_train_final=SP[:,0:int(n_patches_per_img*(1-ratio_val)*n_imgs)]
X_val=X_train_n[int(n_patches_per_img*(1-ratio_val)*n_imgs):,:,:,:]
Y_val=SP[:,int(n_patches_per_img*(1-ratio_val)*n_imgs):]
Val_set=(X_val,np.transpose(Y_val))
model=net.get_LiskowskiSP(patch_size)
model.summary()
early_stopping = EarlyStopping(monitor='val_loss', patience=100)
learning_rate_reduction_loss = ReduceLROnPlateau(monitor='val_loss', 
                                            patience=5, 
                                            verbose=1, 
                                            factor=0.1, 
                                            min_lr=0.00001)
learning_rate_reduction_acc = ReduceLROnPlateau(monitor='val_acc', 
                                            patience=5, 
                                            verbose=1, 
                                            factor=0.1, 
                                            min_lr=0.00001)
history = model.fit(X_train_final, np.transpose(Y_train_final), 
            batch_size, n_epochs,
            validation_split = ratio_val,
            shuffle = True,
            callbacks=[early_stopping, 
learning_rate_reduction_loss, 
learning_rate_reduction_acc],
            verbose=1)
mf.ModelStats(history)
# Save
# serialize model to JSON
model_json = model.to_json()
with open("model_Liskowski_StructPrediction.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model_StructPrediction.h5")
model.save("Liskowski_SP_complet.h5")
print("Saved model to disk")
print('Testing model')
print('...')
X_test = []
Y_test = []
#n_imgs_test=10
n_imgs_test=20
for i in range(n_imgs_test):    
    print("Evaluation on test image ",i, "...")
    # Get Images
    img_path = X_test_folder + os.listdir(X_test_folder)[i]
    gt_path  = Y_test_folder + os.listdir(Y_test_folder)[i]
    val_path = v_test_folder + os.listdir(v_test_folder)[i]
#    # Get Data and Patches
    img, gt, val_mask = mf.getImageData (img_path, gt_path, val_path)
#    X_test, Y_test,my_pos_set,my_neg_set = mf.getTestPatches(img, gt, val_mask, patch_size)
    X_test, Y_test,my_pos_set, my_neg_set  = mf.getTestPatches(img, gt, val_mask, patch_size)
    X_test = X_test.astype('float32')
    X_test = X_test.reshape(X_test.shape[0], 3, patch_size, patch_size)
    prediction=model.predict(X_test,batch_size=32, verbose=1)
#   scores = model.evaluate(X_test, Y_test, verbose=1)
    img_out_viz=mf.Img_reconstruct_SP(my_pos_set, my_neg_set,'DRIVE', prediction)
    img_out_viz=np.reshape(img_out_viz,(img_out_viz.shape[0],img_out_viz.shape[1],1))
    mf.visualize(img_out_viz,"Probmap_%s.png" % i)
    pred_clas=np.empty(prediction.shape[0])
    pred_mean= np.mean(prediction, axis=1)
    pred_clas[pred_mean >=0.50]=1
    pred_clas[pred_mean <0.50]=0
    fpr, tpr, thresholds = roc_curve(Y_test,pred_mean)
    mf.get_metrics(Y_test,pred_clas, 0.50, fpr, tpr, thresholds)
    np.save('prediction_RGB_Balanced_%s' % i, prediction)