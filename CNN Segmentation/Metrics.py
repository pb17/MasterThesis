def GetMetrics(prediction,Y_test,X_test,class_ratio,R_accuracy,R_specificity,R_sensitivity,R_precision,R_aucval)     
    print ("/nConfusion matrix:  Costum threshold (for positive) of " +str(class_ratio))
    y_pred = np.empty((Y_test.shape[0]))
    for i in range(Y_test.shape[0]):
        if prediction[i]>=class_ratio:
           y_pred[i]=1
        else:
            y_pred[i]=0
    confusion = confusion_matrix(Y_test,y_pred)
    print ("Confusion Matrix", confusion)
    if float(np.sum(confusion))!=0:
        R_accuracy.append(float(confusion[0,0]+confusion[1,1])/float(np.sum(confusion)))
    if float(confusion[0,0]+confusion[0,1])!=0:
        R_specificity.append(float(confusion[0,0])/float(confusion[0,0]+confusion[0,1]))   
    if float(confusion[1,1]+confusion[1,0])!=0:
        R_sensitivity.append(float(confusion[1,1])/float(confusion[1,1]+confusion[1,0]))
    if float(confusion[1,1]+confusion[0,1])!=0:
        R_precision.append(float(confusion[1,1])/float(confusion[1,1]+confusion[0,1]))    
    # Area under the ROC curve
    precision, recall, thresholds = precision_recall_curve(Y_test, y_pred)
    R_aucval.append(float(np.trapz(precision,recall)))
    AUC_ROC = roc_auc_score(Y_test y_scores)
    return R_accuracy,R_specificity,R_specificity,R_precision,R_aucval, AUC_ROC
    
 def GetDRIVE()
 # telling where the data is
    X_train_folder = r'C:/Users/Samsung/Desktop/Tese/DataBases/DRIVE/training/images/'
    Y_train_folder = r'C:/Users/Samsung/Desktop/Tese/DataBases/DRIVE/training/1st_manual/'
    v_train_folder = r'C:/Users/Samsung/Desktop/Tese/DataBases/DRIVE/training/mask/'
    X_test_folder  = r'C:/Users/Samsung/Desktop/Tese/DataBases/DRIVE/test/images/'
    Y_test_folder  = r'C:/Users/Samsung/Desktop/Tese/DataBases/DRIVE/test/1st_manual/'
    v_test_folder  = r'C:/Users/Samsung/Desktop/Tese/DataBases/DRIVE/test/mask/'
    n_imgs=20
    img_width=565
    img_heigth=584
    return X_train_folder,Y_train_folder,v_train_folder,X_test_folder,Y_test_folder,v_test_folder, n_imgs
    
def GetSTARE()
 # telling where the data is
    X_train_folder = r'C:/Users/Samsung/Desktop/Tese/DataBases/STARE_1/Traning/image/'
    Y_train_folder = r'C:/Users/Samsung/Desktop/Tese/DataBases/STARE_1/Traning/1st_manual/'
    v_train_folder = r'C:/Users/Samsung/Desktop/Tese/DataBases/STARE_1/Traning/mask/'
    X_test_folder  = r'C:/Users/Samsung/Desktop/Tese/DataBases/STARE_1/Test/images/'
    Y_test_folder  = r'C:/Users/Samsung/Desktop/Tese/DataBases/STARE_1/Test/1st_manual/'
    v_test_folder  = r'C:/Users/Samsung/Desktop/Tese/DataBases/STARE_1/Test/mask/'
    n_imgs=10
    img_width=565
    img_heigth=584
    return X_train_folder,Y_train_folder,v_train_folder,X_test_folder,Y_test_folder,v_test_folder, n_imgs   
    
    