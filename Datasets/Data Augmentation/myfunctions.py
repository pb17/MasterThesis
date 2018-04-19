'''
#===================================================================#
#===================================================================#
'''
import numpy as np
import random

#Keras
from keras.models import Model
from keras.layers import Input, concatenate, Conv2D, MaxPooling2D, UpSampling2D, Reshape, core, Dropout
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras import backend as K
from keras.optimizers import SGD
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
from PIL import Image
import numpy as np
from skimage.segmentation import slic
from skimage.segmentation import mark_boundaries
from matplotlib import pyplot as plt
def GetSTARE():
 # telling where the data is
    X_train_folder = r'C:/Users/Samsung/Desktop/Tese/DataBases/STARE_1/Traning/image/'
    Y_train_folder = r'C:/Users/Samsung/Desktop/Tese/DataBases/STARE_1/Traning/1st_manual/'
    v_train_folder = r'C:/Users/Samsung/Desktop/Tese/DataBases/STARE_1/Traning/mask/'
    X_test_folder  = r'C:/Users/Samsung/Desktop/Tese/DataBases/STARE_1/Test/images/'
    Y_test_folder  = r'C:/Users/Samsung/Desktop/Tese/DataBases/STARE_1/Test/1st_manual/'
    v_test_folder  = r'C:/Users/Samsung/Desktop/Tese/DataBases/STARE_1/Test/mask/'
    n_imgs=10
    img_width=700
    img_heigth=605
    return X_train_folder,Y_train_folder,v_train_folder,X_test_folder,Y_test_folder,v_test_folder, n_imgs,img_width, img_heigth    
def GetDRIVE():
    X_train_folder = r'C:/Users/Samsung/Desktop/Tese/DataBases/DRIVE/training/images/'
    Y_train_folder = r'C:/Users/Samsung/Desktop/Tese/DataBases/DRIVE/training/1st_manual/'
    v_train_folder = r'C:/Users/Samsung/Desktop/Tese/DataBases/DRIVE/training/mask/'
    X_test_folder  = r'C:/Users/Samsung/Desktop/Tese/DataBases/DRIVE/test/images/'
    Y_test_folder  = r'C:/Users/Samsung/Desktop/Tese/DataBases/DRIVE/test/1st_manual/'
    v_test_folder  = r'C:/Users/Samsung/Desktop/Tese/DataBases/DRIVE/test/mask/'
    n_imgs=20
    img_width=565
    img_heigth=584
    return X_train_folder,Y_train_folder,v_train_folder,X_test_folder,Y_test_folder,v_test_folder, n_imgs,img_width, img_heigth  
def getImageData(img_path, gt_path, val_path):    
    img_rgb = Image.open(img_path)
    img = np.asarray(img_rgb, dtype=np.uint8)/255
    gt = Image.open(gt_path);
    gt = np.asarray(gt, dtype=np.uint8)/255
    val_mask = Image.open(val_path);
    val_mask = np.asarray(val_mask, dtype=np.uint8)/255
    return img, gt, val_mask    
def myCNN(patch_size, struct):
    
    model = Sequential()
    
    for i in range(len(struct)):
        if 'A' in struct[i]:
            
            if struct[i][2:] == 'softmax':
                model.add(Activation('softmax'))
                
            elif struct[i][2:] == 'tanh':
                model.add(Activation('tanh'))
                
            elif struct[i][2:] == 'relu':
                model.add(Activation('relu'))

            elif struct[i][2:] == 'sigmoid':
                model.add(Activation('sigmoid'))
                
                    
        elif 'D' in struct[i]:
       
            model.add(Dense(int(struct[i][1:]), kernel_initializer='random_normal', 
                                                bias_initializer='zeros',
                                                kernel_regularizer=regularizers.l2(1e-5),
                                                bias_regularizer = regularizers.l2(1e-5)))
    
        elif 'DP' in struct[i]:
                model.add(Dropout(1/float(struct[i][2:])))
    
        elif 'F' in struct[i]:
            model.add(Flatten())
            
        elif 'C' in struct[i]:
            # Default Value of Kernel size 3
            if i==0:
                model.add(Conv2D(int(struct[i][1:3]),3,data_format = "channels_first",
                                                      padding='valid',
                                                      input_shape=(1,patch_size,patch_size),
                                                      kernel_initializer='random_normal',
                                                      bias_initializer='zeros',
                                                      kernel_regularizer=regularizers.l2(1e-5),
                                                      bias_regularizer = regularizers.l2(1e-5)))
         
            if struct[i][1:].find('_')!=-1:
                dim, kernel_aux= struct[i][1:].split("_")
                model.add(Conv2D(int(dim),int(kernel_aux),data_format = "channels_first",
                                                      padding='valid',
                                                      kernel_initializer='random_normal',
                                                      bias_initializer='zeros',
                                                      kernel_regularizer=regularizers.l2(1e-5),
                                                      bias_regularizer = regularizers.l2(1e-5)))
            
            

            else:
                hey=struct[i][1:]
                model.add(Conv2D(int(struct[i][1:3]),3,data_format = "channels_first",
                                                      padding='valid',
                                                      kernel_initializer='random_normal',
                                                      bias_initializer='zeros',
                                                      kernel_regularizer=regularizers.l2(1e-5),
                                                      bias_regularizer = regularizers.l2(1e-5)))
            
          
           
        elif 'M' in struct[i]: 
            # Default Value of Pool set to  2
            if struct[i][1:]=="":
               model.add(MaxPooling2D(pool_size=2 ,strides=None, data_format = "channels_first"))
            else:
               #model.add(MaxPooling2D(pool_size=int(struct[i][1:]), strides=None, data_format = "channels_first")) 
              model.add(MaxPooling2D((2, 2), strides=None, padding="same", data_format="channels_first", ))
        elif 'I' in struct[i]:
            inputs=Input((int(struct[i][1:]),patch_size,patch_size))
            # Check what channels are used
        elif 'BN' in struct[i]:    
             model.add(BatchNormalization())

        model.compile(loss='binary_crossentropy',
        optimizer=Adam(lr=1e-3, beta_1=0.9, beta_2=0.999, epsilon=1e-5, decay=0),
        metrics=['accuracy']) 
    return model         
def GetDRIVEPRE():
    X_train_folder = r'C:/Users/Samsung/Desktop/Implementações/Wang et al/Original/DataBases/DRIVE/training/images/Pre_pro/'
    Y_train_folder = r'C:/Users/Samsung/Desktop/Implementações/Wang et al/Original/DataBases/DRIVE/training/1st_manual/'
    v_train_folder = r'C:/Users/Samsung/Desktop/Implementações/Wang et al/Original/DataBases/DRIVE/training/mask/'
    X_test_folder  = r'C:/Users/Samsung/Desktop/Implementações/Wang et al/Original/DataBases/DRIVE/test/images/Pre_pro/'
    Y_test_folder  = r'C:/Users/Samsung/Desktop/Implementações/Wang et al/Original/DataBases/DRIVE/test/1st_manual/'
    v_test_folder  = r'C:/Users/Samsung/Desktop/Implementações/Wang et al/Original/DataBases/DRIVE/test/mask/'
    n_imgs=20
    img_width=565
    img_heigth=584
    return X_train_folder,Y_train_folder,v_train_folder,X_test_folder,Y_test_folder,v_test_folder, n_imgs,img_width, img_heigth        
def GetSTAREPRE(): 
    X_train_folder = r'C:/Users/Samsung/Desktop/Implementações/Wang et al/Original/DataBases/STARE_1/Traning/image/Pre pro/'
    Y_train_folder = r'C:/Users/Samsung/Desktop/Implementações/Wang et al/Original/DataBases/STARE_1/Traning/1st_manual/'
    v_train_folder = r'C:/Users/Samsung/Desktop/Implementações/Wang et al/Original/DataBases/STARE_1/Traning/mask/'
    X_test_folder  = r'C:/Users/Samsung/Desktop/Implementações/Wang et al/Original/DataBases/STARE_1/Test/Pre_pro/'
    Y_test_folder  = r'C:/Users/Samsung/Desktop/Implementações/Wang et al/Original/DataBases/STARE_1/Test/1st_manual/'
    v_test_folder  = r'C:/Users/Samsung/Desktop/Implementações/Wang et al/Original/DataBases/STARE_1/Test/mask/'
    n_imgs=20
    img_width=700
    img_heigth=605
    return X_train_folder,Y_train_folder,v_train_folder,X_test_folder,Y_test_folder,v_test_folder, n_imgs, img_width, img_heigth
def get_data_training(DRIVE_train_imgs_original,
                      DRIVE_train_groudTruth,
                      patch_height,
                      patch_width,
                      N_subimgs,
                      inside_FOV):
    train_imgs = DRIVE_train_imgs_original
    train_masks =DRIVE_train_groudTruth #masks always the same
    # visualize(group_images(train_imgs_original[0:20,:,:,:],5),'imgs_train')#.show()  #check original imgs train

    train_imgs = train_imgs[:,:,9:574,:]  #cut bottom and top so now it is 565*565
    train_masks = train_masks[:,:,9:574,:]  #cut bottom and top so now it is 565*565
    data_consistency_check(train_imgs,train_masks)

    #check masks are within 0-1
    assert(np.min(train_masks)==0 and np.max(train_masks)==1)

    print ("\ntrain images/masks shape:")
    print (train_imgs.shape)
    print ("train images range (min-max): " +str(np.min(train_imgs)) +' - '+str(np.max(train_imgs)))
    print ("train masks are within 0-1\n")

    #extract the TRAINING patches from the full images
    patches_imgs_train, patches_masks_train = extract_random(train_imgs,train_masks,patch_height,patch_width,N_subimgs,inside_FOV)
    data_consistency_check(patches_imgs_train, patches_masks_train)

    print ("\ntrain PATCHES images/masks shape:")
    print (patches_imgs_train.shape)
    print ("train PATCHES images range (min-max): " +str(np.min(patches_imgs_train)) +' - '+str(np.max(patches_imgs_train)))

    return patches_imgs_train, patches_masks_train#, patches_imgs_test, patches_masks_test
def getdataok(img_list,gt_list,mask_list):
    img_list=np.asarray(img_list)    
    gt_list=np.asarray(gt_list)
    mask_list=np.asarray(mask_list)
    d1,d2,d3=img_list.shape
    V_img=img_list.reshape(d1,1,d2,d3)
    V_gt=gt_list.reshape(d1,1,d2,d3)
    V_mask=mask_list.reshape(d1,1,d2,d3)
    V_img = V_img.astype('float32')
    V_gt=V_gt.astype('float32')
    V_mask=V_mask.astype('float32')
    return V_img,V_gt,V_mask      
def ModelStats(history): 
    # list all data in history
    print(history.history.keys())
    # summarize history for accuracy
    f, axarr = plt.subplots(2, sharex=True)
    ax = plt.subplot("211")
    ax.set_title("Model Accuracy")
    ax.plot(history.history['acc'])
    ax.plot(history.history['val_acc'])
    ax.set_ylabel('Accuracy')
    
    ax.legend(['train', 'test'], loc='upper left')
    # summarize history for loss
    ax1 = plt.subplot("212")
    ax1.set_title("Model Loss")
    ax1.plot(history.history['loss'])
    ax1.plot(history.history['val_loss'])
    ax1.set_ylabel('Loss')
    ax1.set_xlabel('Epoch')
    ax1.legend(['train', 'test'], loc='upper left')
    return None 
def data_consistency_check(imgs,masks):
    assert(len(imgs.shape)==len(masks.shape))
    assert(imgs.shape[0]==masks.shape[0])
    assert(imgs.shape[2]==masks.shape[2])
    assert(imgs.shape[3]==masks.shape[3])
    assert(masks.shape[1]==1)
    assert(imgs.shape[1]==1 or imgs.shape[1]==3)
def extract_random(full_imgs,full_masks, patch_h,patch_w, N_patches, inside=True):
#    if (N_patches%full_imgs.shape[0] != 0):
#       ( print "N_patches: plase enter a multiple of 20"
#        exit()
#    assert (len(full_imgs.shape)==4 and len(full_masks.shape)==4)  #4D arrays
#    assert (full_imgs.shape[1]==1 or full_imgs.shape[1]==3)  #check the channel is 1 or 3
#    assert (full_masks.shape[1]==1)   #masks only black and white
#    assert (full_imgs.shape[2] == full_masks.shape[2] and full_imgs.shape[3] == full_masks.shape[3])
#    patches = np.empty((N_patches,full_imgs.shape[1],patch_h,patch_w))
#    patches_masks = np.empty((N_patches,full_masks.shape[1],patch_h,patch_w))
#    img_h = full_imgs.shape[2]  #height of the full image
#    img_w = full_imgs.shape[3] #width of the full image
#    # (0,0) in the center of the image
#    patch_per_img = int(N_patches/full_imgs.shape[0])  #N_patches equally divided in the full images
#    print "patches per full image: " +str(patch_per_img)
#    iter_tot = 0   #iter over the total numbe rof patches (N_patches)
#    for i in range(full_imgs.shape[0]):  #loop over the full images
#        k=0
#        while k <patch_per_img:
#            x_center = random.randint(0+int(patch_w/2),img_w-int(patch_w/2))
#            # print "x_center " +str(x_center)
#            y_center = random.randint(0+int(patch_h/2),img_h-int(patch_h/2))
#            # print "y_center " +str(y_center)
#            #check whether the patch is fully contained in the FOV
#            if inside==True:
#                if is_patch_inside_FOV(x_center,y_center,img_w,img_h,patch_h)==False:
#                    continue
#            patch = full_imgs[i,:,y_center-int(patch_h/2):y_center+int(patch_h/2),x_center-int(patch_w/2):x_center+int(patch_w/2)]
#            patch_mask = full_masks[i,:,y_center-int(patch_h/2):y_center+int(patch_h/2),x_center-int(patch_w/2):x_center+int(patch_w/2)]
#            patches[iter_tot]=patch
#            patches_masks[iter_tot]=patch_mask
#            iter_tot +=1   #total
#            k+=1  #per full_img
   return patches, patches_masks
def is_patch_inside_FOV(x,y,img_w,img_h,patch_h):
    x_ = x - int(img_w/2) # origin (0,0) shifted to image center
    y_ = y - int(img_h/2)  # origin (0,0) shifted to image center
    R_inside = 270 - int(patch_h * np.sqrt(2.0) / 2.0) #radius is 270 (from DRIVE db docs), minus the patch diagonal (assumed it is a square #this is the limit to contain the full patch in the FOV
    radius = np.sqrt((x_*x_)+(y_*y_))
    if radius < R_inside:
        return True
    else:
        return False        
def getTrainPatches(image, gt, val_mask, n_patches_per_img, patch_size,class_ratio):
    
    X = np.ndarray([n_patches_per_img,patch_size,patch_size])
    
    R_idx_mask, C_idx_mask = np.where(val_mask==1)
    R_idx_pos,  C_idx_pos  = np.where(gt==1)
    R_idx_neg,  C_idx_neg  = np.where(gt==0)
    
    mask_set = set((R_idx_mask[i], C_idx_mask[i]) for i in range(R_idx_mask.size))
    pos_set = set((R_idx_pos[i], C_idx_pos[i]) for i in range(R_idx_pos.size))
    neg_set = set((R_idx_neg[i], C_idx_neg[i]) for i in range(R_idx_neg.size))
    
    my_pos_set = set((x,y) for (x,y) in pos_set if (x,y) in mask_set)
    my_neg_set = set((x,y) for (x,y) in neg_set if (x,y) in mask_set)
    
    
    n_pos_class = int(n_patches_per_img * class_ratio)
    n_neg_class = n_patches_per_img - n_pos_class
    
    pos_cases = random.sample(my_pos_set, n_pos_class)
    neg_cases = random.sample(my_neg_set, n_neg_class)
    
    for i, pos in enumerate(pos_cases):
        try:
            X[i,:,:] = image[pos[0]-int(np.floor(patch_size/2)):pos[0]+int(np.ceil(patch_size/2)),pos[1]-int(np.floor(patch_size/2)):pos[1]+int(np.ceil(patch_size/2))]
        except:
            print(pos)
    for i, pos in enumerate(neg_cases):  
        try:
            X[i+n_pos_class,:,:] = image[pos[0]-int(np.floor(patch_size/2)):pos[0]+int(np.ceil(patch_size/2)),pos[1]-int(np.floor(patch_size/2)):pos[1]+int(np.ceil(patch_size/2))]
        except:
            print(pos)
    y = np.concatenate((np.repeat(1, n_pos_class), np.repeat(0, n_neg_class)),axis=0)        
    
    return X, y
def getTestPatches(image, gt, val_mask, patch_size):
    
    R_idx_mask, C_idx_mask = np.where(val_mask==1)
    
    X = np.ndarray([R_idx_mask.size,patch_size,patch_size])
    Y = np.ndarray([R_idx_mask.size])
    
    l = patch_size//2
    elig = np.ndarray(val_mask.shape)
    elig[l:elig.shape[0]-l, l:elig.shape[1]-l] = 1
    R_idx_elig, C_idx_elig = np.where(elig==1)
    
    R_idx_pos,  C_idx_pos  = np.where(gt==1)
       
    R_idx_neg,  C_idx_neg  = np.where(gt==0)
    
    mask_set = set((R_idx_mask[i], C_idx_mask[i]) for i in range(R_idx_mask.size))
    elig_set = set((R_idx_elig[i], C_idx_elig[i]) for i in range(R_idx_elig.size)) 
    pos_set = set((R_idx_pos[i], C_idx_pos[i]) for i in range(R_idx_pos.size))
    neg_set = set((R_idx_neg[i], C_idx_neg[i]) for i in range(R_idx_neg.size))
    my_set = set((x,y) for (x,y) in mask_set if (x,y) in elig_set)
    my_pos_set = set((x,y) for (x,y) in pos_set if (x,y) in my_set)
    my_neg_set = set((x,y) for (x,y) in neg_set if (x,y) in my_set)
    N_patches=np.asarray(list(my_set))
    
    for i, pos in enumerate(list(my_pos_set)):
        X[i,:,:] = image[pos[0]-int(np.floor(patch_size/2)):pos[0]+int(np.ceil(patch_size/2)),pos[1]-int(np.floor(patch_size/2)):pos[1]+int(np.ceil(patch_size/2))]
        Y[i] = 1
        
    for i, pos in enumerate(list(my_neg_set)):        
        X[i+len(list(my_pos_set)),:,:] = image[pos[0]-int(np.floor(patch_size/2)):pos[0]+int(np.ceil(patch_size/2)),pos[1]-int(np.floor(patch_size/2)):pos[1]+int(np.ceil(patch_size/2))]
        Y[i+len(list(my_pos_set))] = 0
    
    
    
    return X, Y, N_patches  
def pred_to_imgs(pred, patch_height, patch_width, mode="original"):
    assert (len(pred.shape)==3)  #3D array: (Npatches,height*width,2)
    assert (pred.shape[2]==2 )  #check the classes are 2
    pred_images = np.empty((pred.shape[0],pred.shape[1]))  #(Npatches,height*width)
    if mode=="original":
        for i in range(pred.shape[0]):
            for pix in range(pred.shape[1]):
                pred_images[i,pix]=pred[i,pix,1]
    elif mode=="threshold":
        for i in range(pred.shape[0]):
            for pix in range(pred.shape[1]):
                if pred[i,pix,1]>=0.5:
                    pred_images[i,pix]=1
                else:
                    pred_images[i,pix]=0
    else:
        print ("mode " +str(mode) +" not recognized, it can be 'original' or 'threshold'")
        exit()
    pred_images = np.reshape(pred_images,(pred_images.shape[0],1, patch_height, patch_width))
    return pred_images  
def get_unet(n_ch,patch_height,patch_width):
    inputs = Input(shape=(n_ch,patch_height,patch_width))
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same',data_format='channels_first')(inputs)
    conv1 = Dropout(0.2)(conv1)
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same',data_format='channels_first')(conv1)
    pool1 = MaxPooling2D((2, 2))(conv1)
    #
    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same',data_format='channels_first')(pool1)
    conv2 = Dropout(0.2)(conv2)
    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same',data_format='channels_first')(conv2)
    pool2 = MaxPooling2D((2, 2))(conv2)
    #
    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same',data_format='channels_first')(pool2)
    conv3 = Dropout(0.2)(conv3)
    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same',data_format='channels_first')(conv3)

    up1 = UpSampling2D(size=(2, 2))(conv3)
    up1 = concatenate([conv2,up1],axis=1)
    conv4 = Conv2D(64, (3, 3), activation='relu', padding='same',data_format='channels_first')(up1)
    conv4 = Dropout(0.2)(conv4)
    conv4 = Conv2D(64, (3, 3), activation='relu', padding='same',data_format='channels_first')(conv4)
    #
    up2 = UpSampling2D(size=(2, 2))(conv4)
    up2 = concatenate([conv1,up2], axis=1)
    conv5 = Conv2D(32, (3, 3), activation='relu', padding='same',data_format='channels_first')(up2)
    conv5 = Dropout(0.2)(conv5)
    conv5 = Conv2D(32, (3, 3), activation='relu', padding='same',data_format='channels_first')(conv5)
    #
    conv6 = Conv2D(2, (1, 1), activation='relu',padding='same',data_format='channels_first')(conv5)
    conv6 = core.Reshape((2,patch_height*patch_width))(conv6)
    conv6 = core.Permute((2,1))(conv6)
    ############
    conv7 = core.Activation('softmax')(conv6)

    model = Model(input=inputs, output=conv7)

    # sgd = SGD(lr=0.01, decay=1e-6, momentum=0.3, nesterov=False)
    model.compile(optimizer='sgd', loss='categorical_crossentropy',metrics=['accuracy'])

    return model   
def masks_Unet(masks):
    assert (len(masks.shape)==4)  #4D arrays
    assert (masks.shape[1]==1 )  #check the channel is 1
    im_h = masks.shape[2]
    im_w = masks.shape[3]
    masks = np.reshape(masks,(masks.shape[0],im_h*im_w))
    new_masks = np.empty((masks.shape[0],im_h*im_w,2))
    for i in range(masks.shape[0]):
        for j in range(im_h*im_w):
            if  masks[i,j] == 0:
                new_masks[i,j,0]=1
                new_masks[i,j,1]=0
            else:
                new_masks[i,j,0]=0
                new_masks[i,j,1]=1
    return new_masks    
def get_featuremaps(model, layer_idx, X_batch):
	get_activations = K.function([model.layers[0].input, K.learning_phase()],[model.layers[layer_idx].output,])
	activations = get_activations([X_batch])
	return activations 
def visualize(data,filename):
    assert (len(data.shape)==3) #height*width*channels
    img = None
    if data.shape[2]==1:  #in case it is black and white
        data = np.reshape(data,(data.shape[0],data.shape[1]))
    if np.max(data)>1:
        img = Image.fromarray(data.astype(np.uint8))   #the image is already 0-255
    else:
        img = Image.fromarray((data*255).astype(np.uint8))  #the image is between 0-1
    img.save(filename + '.png')
    return img
def group_images(data,per_row):
    assert data.shape[0]%per_row==0
    assert (data.shape[1]==1 or data.shape[1]==3)
    data = np.transpose(data,(0,2,3,1))  #corect format for imshow
    all_stripe = []
    for i in range(int(data.shape[0]/per_row)):
        stripe = data[i*per_row]
        for k in range(i*per_row+1, i*per_row+per_row):
            stripe = np.concatenate((stripe,data[k]),axis=1)
        all_stripe.append(stripe)
    totimg = all_stripe[0]
    for i in range(1,len(all_stripe)):
        totimg = np.concatenate((totimg,all_stripe[i]),axis=0)
    return totimg        
def pred_only_FOV(data_imgs,data_masks,original_imgs_border_masks):
    assert (len(data_imgs.shape)==4 and len(data_masks.shape)==4)  #4D arrays
    assert (data_imgs.shape[0]==data_masks.shape[0])
    assert (data_imgs.shape[2]==data_masks.shape[2])
    assert (data_imgs.shape[3]==data_masks.shape[3])
    assert (data_imgs.shape[1]==1 and data_masks.shape[1]==1)  #check the channel is 1
    height = data_imgs.shape[2]
    width = data_imgs.shape[3]
    new_pred_imgs = []
    new_pred_masks = []
    for i in range(data_imgs.shape[0]):  #loop over the full images
        for x in range(width):
            for y in range(height):
                if inside_FOV_DRIVE(i,x,y,original_imgs_border_masks)==True:
                    new_pred_imgs.append(data_imgs[i,:,y,x])
                    new_pred_masks.append(data_masks[i,:,y,x])
    new_pred_imgs = np.asarray(new_pred_imgs)
    new_pred_masks = np.asarray(new_pred_masks)
    return new_pred_imgs, new_pred_masks       
def inside_FOV_DRIVE(i, x, y, DRIVE_masks):
    assert (len(DRIVE_masks.shape)==4)  #4D arrays
    assert (DRIVE_masks.shape[1]==1)  #DRIVE masks is black and white
    # DRIVE_masks = DRIVE_masks/255.  #NOOO!! otherwise with float numbers takes forever!!

    if (x >= DRIVE_masks.shape[3] or y >= DRIVE_masks.shape[2]): #my image bigger than the original
        return False

    if (DRIVE_masks[i,0,y,x]>0):  #0==black pixels
        # print DRIVE_masks[i,0,y,x]  #verify it is working right
        return True
    else:
        return False    
def Rfin(feature_map):
    VectorFeatures=np.empty(feature_map.size)
    aux=0;
    d1=feature_map.shape
    for k in range (d1[2]):
       for j in range(d1[1]):
          for i in range (d1[0]):
              VectorFeatures[aux]=feature_map[i,k,j]
              aux=aux+1
              
    return VectorFeatures      
def showsuperpixel(All_img):
   segments = slic(All_img[0,0], n_segments = 575, sigma = 5)
   fig = plt.figure("Superpixels -- %d segments" % (575))
   ax = fig.add_subplot(1, 1, 1)
   ax.imshow(mark_boundaries(All_img[0,0], segments))
   plt.axis("off")
   # show the plots
   plt.show()
   return None 
def get_data_training_deepFeatures(DRIVE_train_imgs_original,
                                      DRIVE_train_groudTruth,
                                                Imgs_to_test,
                                                patch_height,
                                                patch_width,    
                                                stride_height, stride_width):
    test_imgs = DRIVE_train_imgs_original
    test_masks =DRIVE_train_groudTruth
    test_imgs = test_imgs[0:Imgs_to_test,:,:,:]
    test_masks = test_masks[0:Imgs_to_test,:,:,:]
    test_imgs = paint_border_overlap(test_imgs, patch_height, patch_width, stride_height, stride_width)
    test_masks = paint_border_overlap(test_masks, patch_height, patch_width, stride_height, stride_width)
    #check masks are within 0-1
    assert(np.max(test_masks)==1  and np.min(test_masks)==0)

    print ("\ntest images shape:")
    print (test_imgs.shape)
    print ("\ntest mask shape:")
    print (test_masks.shape)
    print ("test images range (min-max): " +str(np.min(test_imgs)) +' - '+str(np.max(test_imgs)))
    print ("test masks are within 0-1\n")
    patches_imgs_train = extract_ordered_overlap(test_imgs,patch_height,patch_width,stride_height,stride_width)
    patches_mask_train = extract_ordered_overlap(test_imgs,patch_height,patch_width,stride_height,stride_width)
    
    return patches_imgs_train, patches_mask_train     
def extract_ordered_overlap(full_imgs, patch_h, patch_w,stride_h,stride_w):
    assert (len(full_imgs.shape)==4)  #4D arrays
    assert (full_imgs.shape[1]==1 or full_imgs.shape[1]==3)  #check the channel is 1 or 3
    img_h = full_imgs.shape[2]  #height of the full image
    img_w = full_imgs.shape[3] #width of the full image
    assert ((img_h-patch_h)%stride_h==0 and (img_w-patch_w)%stride_w==0)
    N_patches_img = ((img_h-patch_h)//stride_h+1)*((img_w-patch_w)//stride_w+1)  #// --> division between integers
    N_patches_tot = N_patches_img*full_imgs.shape[0]
    print ("Number of patches on h : " +str(((img_h-patch_h)//stride_h+1)))
    print ("Number of patches on w : " +str(((img_w-patch_w)//stride_w+1)))
    print ("number of patches per image: " +str(N_patches_img) +", totally for this dataset: " +str(N_patches_tot))
    patches = np.empty((N_patches_tot,full_imgs.shape[1],patch_h,patch_w))
    iter_tot = 0   #iter over the total number of patches (N_patches)
    for i in range(full_imgs.shape[0]):  #loop over the full images
        for h in range((img_h-patch_h)//stride_h+1):
            for w in range((img_w-patch_w)//stride_w+1):
                patch = full_imgs[i,:,h*stride_h:(h*stride_h)+patch_h,w*stride_w:(w*stride_w)+patch_w]
                patches[iter_tot]=patch
                iter_tot +=1   #total
    assert (iter_tot==N_patches_tot)
    return patches  #array with all the full_imgs divided in patches    
def paint_border_overlap(full_imgs, patch_h, patch_w, stride_h, stride_w):
    assert (len(full_imgs.shape)==4)  #4D arrays
    assert (full_imgs.shape[1]==1 or full_imgs.shape[1]==3)  #check the channel is 1 or 3
    img_h = full_imgs.shape[2]  #height of the full image
    img_w = full_imgs.shape[3] #width of the full image
    leftover_h = (img_h-patch_h)%stride_h  #leftover on the h dim
    leftover_w = (img_w-patch_w)%stride_w  #leftover on the w dim
    if (leftover_h != 0):  #change dimension of img_h
        print ("\nthe side H is not compatible with the selected stride of " +str(stride_h))
        print ("img_h " +str(img_h) + ", patch_h " +str(patch_h) + ", stride_h " +str(stride_h))
        print ("(img_h - patch_h) MOD stride_h: " +str(leftover_h))
        print ("So the H dim will be padded with additional " +str(stride_h - leftover_h) + " pixels")
        tmp_full_imgs = np.zeros((full_imgs.shape[0],full_imgs.shape[1],img_h+(stride_h-leftover_h),img_w))
        tmp_full_imgs[0:full_imgs.shape[0],0:full_imgs.shape[1],0:img_h,0:img_w] = full_imgs
        full_imgs = tmp_full_imgs
    if (leftover_w != 0):   #change dimension of img_w
        print ("the side W is not compatible with the selected stride of " +str(stride_w))
        print ("img_w " +str(img_w) + ", patch_w " +str(patch_w) + ", stride_w " +str(stride_w))
        print ("(img_w - patch_w) MOD stride_w: " +str(leftover_w))
        print ("So the W dim will be padded with additional " +str(stride_w - leftover_w) + " pixels")
        tmp_full_imgs = np.zeros((full_imgs.shape[0],full_imgs.shape[1],full_imgs.shape[2],img_w+(stride_w - leftover_w)))
        tmp_full_imgs[0:full_imgs.shape[0],0:full_imgs.shape[1],0:full_imgs.shape[2],0:img_w] = full_imgs
        full_imgs = tmp_full_imgs
    print ("new full images shape: \n" +str(full_imgs.shape))
    return full_imgs
def test(activations02):
    VectorFeatures=np.empty(activations02.size)
    aux=0;
    d1=activations02.shape
    for k in range (d1[2]):
      for j in range(d1[1]):
          for i in range (d1[0]):
           VectorFeatures[aux]=activations02[i,k,j]
           aux=aux+1
    return  VectorFeatures
    


    
def ZCAW(X_train_n, patch_size):

    def zca_whitening_matrix(X):

     sigma = np.cov(X, rowvar=True) 
     U,S,V = np.linalg.svd(sigma)
     epsilon = 1e-5
     ZCAMatrix = np.dot(U, np.dot(np.diag(1.0/np.sqrt(S + epsilon)), U.T))
    
     return ZCAMatrix



     OutZCAMatrix=[]
     for i in range(X_train_n.shape[0]): 
        X=X_train_n[i,0]
        ZCAMatrix = zca_whitening_matrix(X) # get ZCAMatrix
        ZCAMatrix # [5 x 5] matrix
        xZCAMatrix = np.dot(ZCAMatrix, X) # project X onto the ZCAMatrix
        OutZCAMatrix.append(xZCAMatrix)
     
     ZCAMatrix_1=np.asarray(OutZCAMatrix)
     ZCAMatrix_1= ZCAMatrix_1.reshape(X_train_n.shape[0], 1, patch_size, patch_size)     
    
     return ZCAMatrix_1