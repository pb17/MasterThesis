# -*- coding: utf-8 -*-

"""
Melinscak - CNN Segmentation 
"""
from keras import regularizers
from keras.optimizers import Adam, SGD, Adagrad, RMSprop
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
from PIL import Image
import numpy as np
import fuctionsrep as mf
import extract_patches as ex
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
print('Melinscak - Green Channel and Pre processed comparson')
print('...')
# parameters controlling the training procedure
n_patches_per_img = 10500
patch_size = 25
class_ratio = 0.20
n_epochs = 150
ratio_val = 0.25
batch_size = 32
#X_train_folder,Y_train_folder,v_train_folder,X_test_folder,Y_test_folder,v_test_folder, n_imgs,img_width, img_heigth1=mf.GetSTAREPRE()
X_train_folder,Y_train_folder,v_train_folder,X_test_folder,Y_test_folder,v_test_folder, n_imgs,img_width, img_heigth1=mf.GetDRIVEPRE()
X_train  = np.ndarray([n_patches_per_img*n_imgs,patch_size,patch_size])
Y_train = np.ndarray([n_patches_per_img*n_imgs])
for i in range(15):
    img_path = X_train_folder + os.listdir(X_train_folder)[i]
    gt_path  = Y_train_folder + os.listdir(Y_train_folder)[i]
    val_path = v_train_folder + os.listdir(v_train_folder)[i]    
    img, gt, val_mask = mf.getImageData (img_path, gt_path, val_path)
   # img, gt, val_mask = mf.getImageDataGreen(img_path, gt_path, val_path)
    X_train[i*n_patches_per_img:(i+1)*n_patches_per_img,:,:], Y_train[i*n_patches_per_img:(i+1)*n_patches_per_img] = mf.getTrainPatches(img, gt, val_mask, n_patches_per_img, patch_size, class_ratio)

X_train_n = X_train.astype('float32')
X_train_n = X_train_n.reshape(X_train_n.shape[0], 1, patch_size, patch_size)
X_train_final=X_train_n[0:int(n_patches_per_img*(1-ratio_val)*n_imgs),:,:,:]
Y_train_final=Y_train[0:int(n_patches_per_img*(1-ratio_val)*n_imgs)]
X_val=X_train_n[int(n_patches_per_img*(1-ratio_val)*n_imgs):,:,:,:]
Y_val=Y_train[int(n_patches_per_img*(1-ratio_val)*n_imgs):]
Val_set=(X_val,Y_val)
model = Sequential()
model.add(Conv2D(65,6,data_format = "channels_first",activation=None,
                                                      padding='same',
                                                      input_shape=(1,patch_size,patch_size),
                                                      kernel_initializer='random_normal',
                                                      bias_initializer='zeros',
                                                      kernel_regularizer=regularizers.l2(1e-5),
                                                      bias_regularizer = regularizers.l2(1e-5),name='C01'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=None,  name='MP01',data_format = "channels_first"))
model.add(Conv2D(26,5,data_format = "channels_first",activation=None,
                                                      padding='same',
                                                      input_shape=(1,patch_size,patch_size),
                                                      kernel_initializer='random_normal',
                                                      bias_initializer='zeros',
                                                      kernel_regularizer=regularizers.l2(1e-5),
                                                      bias_regularizer = regularizers.l2(1e-5),name='C02'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=None,  name='MP02',data_format = "channels_first"))
model.add(Conv2D(10,5,data_format = "channels_first",activation=None,
                                                      padding='same',
                                                      input_shape=(1,patch_size,patch_size),
                                                      kernel_initializer='random_normal',
                                                      bias_initializer='zeros',
                                                      kernel_regularizer=regularizers.l2(1e-5),
                                                      bias_regularizer = regularizers.l2(1e-5),name='C04'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=None,  name='MP04',data_format = "channels_first"))
model.add(Conv2D(4,2,data_format = "channels_first",activation=None,
                                                      padding='same',
                                                      input_shape=(1,patch_size,patch_size),
                                                      kernel_initializer='random_normal',
                                                      bias_initializer='zeros',
                                                      kernel_regularizer=regularizers.l2(1e-5),
                                                      bias_regularizer = regularizers.l2(1e-5),name='C05'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=None,  name='MP05',data_format = "channels_first"))
model.add(Flatten())
model.add(Dense(100, kernel_initializer='random_normal', 
                                                bias_initializer='zeros',
                                                kernel_regularizer=regularizers.l2(1e-5),
                                                bias_regularizer = regularizers.l2(1e-5)))
model.add(Activation('sigmoid'))
model.add(Dense(1, kernel_initializer='random_normal', 
                                                bias_initializer='zeros',
                                                kernel_regularizer=regularizers.l2(1e-5)))
model.add(Activation('sigmoid'))
model.compile(loss='binary_crossentropy',
    optimizer=Adam(lr=1e-3, beta_1=0.9, beta_2=0.999, epsilon=1e-5, decay=0),
    metrics=['accuracy'])      
model.summary()
early_stopping = EarlyStopping(monitor='val_loss', patience=100)
learning_rate_reduction_loss = ReduceLROnPlateau(monitor='val_loss', 
                                            patience=5, 
                                            verbose=1, 
                                            factor=0.01, 
                                            min_lr=0.00001)
learning_rate_reduction_acc = ReduceLROnPlateau(monitor='val_acc', 
                                            patience=5, 
                                            verbose=1, 
                                            factor=0.01, 
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
model_json = model.to_json()
with open("model_Melinscak_Net_STARE.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model_Melinscak_Net_STARE.h5")
model.save("model_Melinscak_STARE.h5")
print("Saved model to disk")
print('Testing model')
print('...')
X_test = []
Y_test = []
n_imgs_test=20

for i in range(n_imgs_test):    
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
    prediction=model.predict(X_test,batch_size=32, verbose=1)
 #   scores = model.evaluate(X_test, Y_test, verbose=1)
    X_test, Y_test,my_pos_set,my_neg_set = mf.getTestPatches(img, gt, val_mask, patch_size)
    out_img=mf.Image_reconstruct(my_pos_set, my_neg_set, prediction, 'DRIVE','prob')
    img_out_viz=np.reshape(out_img,(out_img.shape[0],out_img.shape[1],1))
    mf.visualize(img_out_viz,"predicitons_%s_DRIVE_V2.png" % i)
    pred_clas=np.empty(prediction.shape[0])
    pred_clas[prediction[:,0] >=0.5]=1
    pred_clas[prediction[:,0] <0.5 ]=0
    fpr, tpr, thresholds = roc_curve(Y_test,pred_clas)
    mf.get_metrics(Y_test,pred_clas, 0.5, fpr, tpr, thresholds)
    np.save('Melinscak_perd_img_%s_DRIVE_v2' %i,prediction)