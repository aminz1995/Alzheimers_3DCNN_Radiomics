#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Amin Zarei (Gmail: aminz1995@gmail.com)
"""
#------------------------------------------------------------------------------
# import global modules
import os
import numpy as np
import itertools
import keras.backend as K
from scipy.ndimage import zoom
from sklearn.preprocessing import binarize
from keras.layers.convolutional import MaxPooling3D, Conv3D, UpSampling3D
from keras.layers import Dense, Flatten, Dropout, BatchNormalization, Softmax, Input, LeakyReLU, Reshape, Activation, concatenate
from keras import layers, regularizers
from sklearn.utils import shuffle
from keras.models import Model, load_model, Sequential
from keras import callbacks
from sklearn.svm import SVC
from keras.utils import np_utils, plot_model
from sklearn.model_selection import train_test_split
from sklearn import metrics
from keras.optimizers import Adam, RMSprop, Adadelta
import matplotlib.pyplot as plt
from keras.applications.mobilenet import MobileNet
from keras.applications.nasnet import NASNetMobile, NASNetLarge
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.utils import compute_sample_weight, compute_class_weight
from keras.utils import np_utils
import nipype.interfaces.fsl as fsl
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
import SimpleITK as sitk
from keras.layers import GlobalAveragePooling3D
import scipy.io as sio
from sklearn.preprocessing import MinMaxScaler, Normalizer, MaxAbsScaler, RobustScaler, StandardScaler
from sklearn.model_selection import KFold, LeaveOneOut, GridSearchCV
from sklearn import decomposition
import pandas as pd
import pickle
from scipy import stats
from sklearn.utils import class_weight
from keras import initializers, regularizers
import re, datetime

#------------------------------------------------------------------------------
# configs
ROI_1 = '/Hippocampus/'
ROI_2 = '/Amygdala/'
    
structures_1 = ['L_Hipp', 'R_Hipp']
structures_2 = ['L_Amyg', 'R_Amyg']
 
Size_1 = [16, 40, 24]
Size_2 = [20, 20, 20]

type_input = 'compose-Hippo-Amyg'     #  <compose-Hippo-Amyg>, <just-Hippo>, <just-Amyg>
type_classifier = 'just-Radiomics'  # <compose-CNN-Radiomics>, <just-CNN>, <just-Radiomics>       

groups = ['AD', 'CN']
   
radiomics_path_left_1 = './features' + ROI_1 + 'left/'
radiomics_path_right_1 = './features' + ROI_1 + 'right/' 

radiomics_path_left_2 = './features' + ROI_2 + 'left/'
radiomics_path_right_2 = './features' + ROI_2 + 'right/'  

A_path = './ADNI/BrainExtracted/FreeSurfer_Cross-Sectional_Processing_brainmask/'+groups[0]+'_FS_B_mask_CS/'
B_path = './ADNI/BrainExtracted/FreeSurfer_Cross-Sectional_Processing_brainmask/'+groups[1]+'_FS_B_mask_CS/'


svm_params = {"kernel": ['rbf', 'poly'],
                  "C": [1.0, 10.0, 100.0, 500.0],
                  "degree": [2],
                  "coef0": [1.0]}

        
kk = 5 # eg choose 10 for 10-fold cross validation

plt_model = False 
opt = Adam
learning_rate = 0.001 
Loss_function = 'binary_crossentropy'
epochs = 250
batch = 40  
rs = 2020  # random-seed

#------------------------------------------------------------------------------
# Define Functions : ########################################################## 

def prepare(pkl_file):

    features = []
    names = []
    for i in range(np.shape(pkl_file)[0]):
        
        f = []
        n = []
        for featureName in pkl_file[i].keys():
            
            f.append(pkl_file[i][featureName])
            n.append(featureName)
            
        features.append(f[22:])
        names.append(n[22:])
    
    output = np.array(features)
    
    return output, names[0]

############################################################################### 
    
def load_names(root_path, structures):
    
    brain_images = []
    left_masks = []
    right_masks = []

    total_subjects = len(os.listdir(root_path))
    print('total subjects:', total_subjects)
    subjects = os.listdir(root_path)
    
    for i in range(np.shape(subjects)[0]):
        subject_images_dir = root_path + subjects[i] + "/"
        subject_images = os.listdir(subject_images_dir)
        for folder in subject_images:
            path = subject_images_dir + folder + "/"
            
            for subdir, dirs, files in os.walk(path):
                for file in files:
                    filepath = subdir + os.sep + file
                    
                    if filepath.endswith("brainmask.nii") and folder in filepath:                      
                        brain_images.append(filepath)
                        
                    elif filepath.endswith("brainmask-"+structures[0]+"_corr_Cropped.mat") and folder in filepath:                       
                        left_masks.append(filepath)
                        
                    elif filepath.endswith("brainmask-"+structures[1]+"_corr_Cropped.mat") and folder in filepath:                       
                        right_masks.append(filepath)


    return brain_images, left_masks, right_masks     


###############################################################################

def load_and_prepare_roi(left_roi, right_roi, structures, label, size):
    
    left_images = []
    right_images = []
    Labels = []
    
    if label == "AD":
        l = 0
    elif label == "CN":
        l = 1
    elif label == "MCI":
        l = 2       
        
    for i in range(np.shape(left_roi)[0]):
        
        left = sio.loadmat(left_roi[i])[structures[0]]
        right = sio.loadmat(right_roi[i])[structures[1]]
        
     
        left = zoom(left, (size[0]/left.shape[0], size[1]/left.shape[1], size[2]/left.shape[2]))
        right = zoom(right, (size[0]/right.shape[0], size[1]/right.shape[1], size[2]/right.shape[2]))
       
        left = np.array(left).astype('float32')
        left = (left - np.min(left))/(np.max(left) - np.min(left))
        
        right = np.array(right).astype('float32')
        right = (right - np.min(right))/(np.max(right) - np.min(right)) 
       
        left_images.append(left)
        right_images.append(right)
        Labels.append(l)
        
        print("load " + str(i+1) + " of " + str(np.shape(left_roi)[0]))

    return np.array(left_images), np.array(right_images), np.array(Labels)


###############################################################################

def create_convolution_layers(input_img, side):
    model = Conv3D(8, (3, 3, 3), strides=(2, 2, 2), padding='same', activation='relu', kernel_initializer='glorot_normal', name='conv1_'+side)(input_img)
    model = Dropout(0.25, name='drop1_'+side)(model)
    
    model = Conv3D(16, (3, 3, 3), strides=(2, 2, 2), padding='same', activation='relu', kernel_initializer='glorot_normal', name='conv2_'+side)(model)
    model = Dropout(0.25, name='drop2_'+side)(model)
        
    model = Conv3D(32, (3, 3, 3), strides=(2, 2, 2), padding='same', activation='relu', kernel_initializer='glorot_normal', name='conv3_'+side)(model)
    model = BatchNormalization()(model)   
    model = Dropout(0.3, name='drop3_'+side)(model)
  
    return model


###############################################################################   
   
def kfold_index_generator(k, n):
           
    kf = KFold(n_splits=k, random_state=None, shuffle=False)
    idx_x = [] 
    idx_y = []
    for x, y in kf.split(range(n)):
        
        idx_x.append(x)
        idx_y.append(y)
        
    return idx_x, idx_y
    
###############################################################################
def extract_ages_sex(addresses, metadata):
    Ages, Sex = [], []
    date_format1 = "%m/%d/%Y"
    date_format2 = "%Y-%m-%d"
    for i in range(np.shape(addresses)[0]):
        D = addresses[i].split('/')[-3]
        S = addresses[i].split('/')[-5]  
        
        match = re.search('\d{4}-\d{2}-\d{2}', str(D))
        
        date2 = datetime.datetime.strptime(match.group(), date_format2).date()
        date1 = date2.strftime(date_format1)
        date1 = date1.split("/")
        date1[0] = str(int(date1[0]))
        date1 = str(date1[0]+"/"+date1[1]+"/"+date1[2])
                    
        x = np.where(metadata["Acq Date"]==date1)[0]
        y = np.where(metadata["Subject"]==S)[0]
        
        for element in x:
            if element in y:
                z = element
                
        Ages.append(metadata['Age'][z]/100)
        
        if metadata['Sex'][z] == 'F':
            sex = -1
        if metadata['Sex'][z] == 'M':
            sex = 1
        Sex.append(sex)
        
    return np.expand_dims((Ages), axis=1), np.expand_dims((Sex), axis=1)

###############################################################################
    
def perform_t_test(features1, features2):
   T, P = [], []        
   for i in range(np.shape(features1)[1]):
       a = features1[:, i]
       b = features2[:, i]
       a = np.expand_dims(a, axis=1)
       b = np.expand_dims(b, axis=1)
       t, p = stats.ttest_ind(a, b, axis=0, equal_var=False)
       T.append(t)
       P.append(p)
        
   return T, P

###############################################################################

def plot_history(network_history):
    
    plt.plot(network_history.history['acc'])
    plt.plot(network_history.history['val_acc'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()
    plt.figure()

    # Plot training & validation loss values
    plt.plot(network_history.history['loss'])
    plt.plot(network_history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()

###############################################################################

def plot_confusion_matrix(matrix, labels,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        matrix = matrix.astype('float') / matrix.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    #print(matrix)
    plt.figure()
    plt.imshow(matrix, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(labels))
    plt.xticks(tick_marks, labels) #, rotation=45)
    plt.yticks(tick_marks, labels)

    fmt = '.2f' if normalize else 'd'
    thresh = matrix.max() / 2.
    for i, j in itertools.product(range(matrix.shape[0]), range(matrix.shape[1])):
        plt.text(j, i, format(matrix[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if matrix[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    
###############################################################################
   
   

if __name__ == "__main__":
    
       
    A_brain_images, A_left_rois_1, A_right_rois_1 = load_names(A_path, structures_1)  
    B_brain_images, B_left_rois_1, B_right_rois_1 = load_names(B_path, structures_1)

    A_brain_images, A_left_rois_2, A_right_rois_2 = load_names(A_path, structures_2)  
    B_brain_images, B_left_rois_2, B_right_rois_2 = load_names(B_path, structures_2)
    
    
   
    A_left_1, A_right_1, A_Labels_1 = load_and_prepare_roi(A_left_rois_1, A_right_rois_1, structures_1, label=groups[0], size=Size_1)
    B_left_1, B_right_1, B_Labels_1 = load_and_prepare_roi(B_left_rois_1, B_right_rois_1, structures_1, label=groups[1], size=Size_1)
          
    A_left_2, A_right_2, A_Labels_2 = load_and_prepare_roi(A_left_rois_2, A_right_rois_2, structures_2, label=groups[0], size=Size_2)
    B_left_2, B_right_2, B_Labels_2 = load_and_prepare_roi(B_left_rois_2, B_right_rois_2, structures_2, label=groups[1], size=Size_2)
          
    
    
    # A_brain = load_and_prepare_brain(A_brain_images)
    # B_brain = load_and_prepare_brain(B_brain_images)
    
    
    ############################################################################
    
    A_metadata = pd.read_csv("./matadata/"+ groups[0] + '_MetaData.csv')
    B_metadata = pd.read_csv("./matadata/"+ groups[1] + '_MetaData.csv')

    A_Ages, A_Sex = extract_ages_sex(A_brain_images, A_metadata)
    B_Ages, B_Sex = extract_ages_sex(B_brain_images, B_metadata)

    A_Ages_Sex = np.concatenate((A_Ages, A_Sex), axis=1)
    B_Ages_Sex = np.concatenate((B_Ages, B_Sex), axis=1)
    
    ############################################################################
    

    A_pkl_left_1 = pickle.load(open(radiomics_path_left_1+groups[0]+"_features.pkl","rb"))
    B_pkl_left_1 = pickle.load(open(radiomics_path_left_1+groups[1]+"_features.pkl","rb"))
 
   
    A_pkl_right_1 = pickle.load(open(radiomics_path_right_1+groups[0]+"_features.pkl","rb"))
    B_pkl_right_1 = pickle.load(open(radiomics_path_right_1+groups[1]+"_features.pkl","rb"))
    
    
    A_pkl_left_2 = pickle.load(open(radiomics_path_left_2+groups[0]+"_features.pkl","rb"))
    B_pkl_left_2 = pickle.load(open(radiomics_path_left_2+groups[1]+"_features.pkl","rb"))
 
    
    A_pkl_right_2 = pickle.load(open(radiomics_path_right_2+groups[0]+"_features.pkl","rb"))
    B_pkl_right_2 = pickle.load(open(radiomics_path_right_2+groups[1]+"_features.pkl","rb"))
    
    
    
    A_features_left_1, feature_names = prepare(A_pkl_left_1)
    B_features_left_1, feature_names = prepare(B_pkl_left_1)
    
    A_features_right_1, feature_names = prepare(A_pkl_right_1)
    B_features_right_1, feature_names = prepare(B_pkl_right_1)
    
    
    A_features_left_2, feature_names = prepare(A_pkl_left_2)
    B_features_left_2, feature_names = prepare(B_pkl_left_2)

    A_features_right_2, feature_names = prepare(A_pkl_right_2)
    B_features_right_2, feature_names = prepare(B_pkl_right_2)

    
    #--------------------------------------------------------------------------
    # minn = np.min((np.shape(AD_Labels_1)[0], np.shape(CN_Labels_1)[0]))
    
    
    # T, P = perform_t_test(AD_features_left_1[0:minn], CN_features_left_1[0:minn])
    # index_left_1 = np.where(np.array(P) < 0.05 )[0]
    
    # T, P = perform_t_test(AD_features_right_1[0:minn], CN_features_right_1[0:minn])
    # index_right_1 = np.where(np.array(P) < 0.05 )[0]
    
    
    
    # minn = np.min((np.shape(AD_Labels_2)[0], np.shape(CN_Labels_2)[0]))
    
    # T, P = perform_t_test(AD_features_left_2[0:minn], CN_features_left_2[0:minn])
    # index_left_2 = np.where(np.array(P) < 0.05 )[0]
    
    # T, P = perform_t_test(AD_features_right_2[0:minn], CN_features_right_2[0:minn])
    # index_right_2 = np.where(np.array(P) < 0.05 )[0]
    
    
    #[:, index_right_1]
    
    #--------------------------------------------------------------------------
    
    A_features_1 = np.concatenate((A_features_left_1, A_features_right_1), axis=1)
    B_features_1 = np.concatenate((B_features_left_1, B_features_right_1), axis=1)

    A_features_2 = np.concatenate((A_features_left_2, A_features_right_2), axis=1)
    B_features_2 = np.concatenate((B_features_left_2, B_features_right_2), axis=1)

    #--------------------------------------------------------------------------
    
    x_1 = np.shape(A_features_1)[0]
    y_1 = np.shape(B_features_1)[0]
    
    features_1 = np.concatenate((A_features_1, B_features_1), axis=0)
    for v in range(np.shape(features_1)[1]):
        features_1[:,v] = (features_1[:,v]-np.min(features_1[:,v]))/(np.max(features_1[:,v])-np.min(features_1[:,v]))  
       
    A_features_1 = features_1[0:x_1]
    B_features_1 = features_1[x_1:]
       
    A_features_1 = np.delete(A_features_1, np.where(np.isnan(A_features_1)==True)[1], axis=1)
    B_features_1 = np.delete(B_features_1, np.where(np.isnan(B_features_1)==True)[1], axis=1)
    
    
    
    x_2 = np.shape(A_features_2)[0]
    y_2 = np.shape(B_features_2)[0] 

    features_2 = np.concatenate((A_features_2, B_features_2), axis=0)
    for v in range(np.shape(features_2)[1]):
        features_2[:,v] = (features_2[:,v]-np.min(features_2[:,v]))/(np.max(features_2[:,v])-np.min(features_2[:,v]))  
               
    A_features_2 = features_2[0:x_2]
    B_features_2 = features_2[x_2:]
    
    A_features_2 = np.delete(A_features_2, np.where(np.isnan(A_features_2)==True)[1], axis=1)
    B_features_2 = np.delete(B_features_2, np.where(np.isnan(B_features_2)==True)[1], axis=1)
    
    
    ############################################################################
    
    
    idx_x1, idx_y1 = kfold_index_generator(k=kk, n=np.shape(A_brain_images)[0])
    idx_x2, idx_y2 = kfold_index_generator(k=kk, n=np.shape(B_brain_images)[0])
    
    acc_kfold_classifier1 = []
    auc_kfold_classifier1 = []
    
    acc_kfold_classifier2 = []
    auc_kfold_classifier2 = []
    
    acc_kfold_classifier3 = []
    auc_kfold_classifier3 = []
    
    acc_kfold_classifier_mv = []
    auc_kfold_classifier_mv = []
    
    acc_kfold_classifier4 = []
    auc_kfold_classifier4 = []
  
    
    for i in range(0, kk):
  
        K.clear_session()
        
        A_train_left_1, A_train_right_1 = A_left_1[idx_x1[i]], A_right_1[idx_x1[i]]
        B_train_left_1, B_train_right_1 = B_left_1[idx_x2[i]], B_right_1[idx_x2[i]]
        
        A_train_left_2, A_train_right_2 = A_left_2[idx_x1[i]], A_right_2[idx_x1[i]]
        B_train_left_2, B_train_right_2 = B_left_2[idx_x2[i]], B_right_2[idx_x2[i]]


        A_test_left_1, A_test_right_1 = A_left_1[idx_y1[i]], A_right_1[idx_y1[i]]
        B_test_left_1, B_test_right_1 = B_left_1[idx_y2[i]], B_right_1[idx_y2[i]]
               
        A_test_left_2, A_test_right_2 = A_left_2[idx_y1[i]], A_right_2[idx_y1[i]]
        B_test_left_2, B_test_right_2 = B_left_2[idx_y2[i]], B_right_2[idx_y2[i]]
               
        
        ############################################################################
        
        A_Age_Sex_train = A_Ages_Sex[idx_x1[i]]
        B_Age_Sex_train = B_Ages_Sex[idx_x2[i]]
        
        A_Age_Sex_test = A_Ages_Sex[idx_y1[i]]
        B_Age_Sex_test = B_Ages_Sex[idx_y2[i]]
        
        ############################################################################  
        
        train_cnn_left_1 = np.concatenate((A_train_left_1, B_train_left_1), axis=0) 
        train_cnn_left_2 = np.concatenate((A_train_left_2, B_train_left_2), axis=0) 
        train_cnn_right_1 = np.concatenate((A_train_right_1, B_train_right_1), axis=0)
        train_cnn_right_2 = np.concatenate((A_train_right_2, B_train_right_2), axis=0)
       
        test_cnn_left_1 = np.concatenate((A_test_left_1, B_test_left_1), axis=0)
        test_cnn_left_2 = np.concatenate((A_test_left_2, B_test_left_2), axis=0)      
        test_cnn_right_1 = np.concatenate((A_test_right_1, B_test_right_1), axis=0)
        test_cnn_right_2 = np.concatenate((A_test_right_2, B_test_right_2), axis=0)
        
        ############################################################################

        train_age_sex = np.concatenate((A_Age_Sex_train, B_Age_Sex_train), axis=0)
        test_age_sex = np.concatenate((A_Age_Sex_test, B_Age_Sex_test), axis=0)
        
        ############################################################################
        
        train_radiomics_1 = np.concatenate((A_features_1[idx_x1[i]], B_features_1[idx_x2[i]]), axis=0)    
        train_radiomics_2 = np.concatenate((A_features_2[idx_x1[i]], B_features_2[idx_x2[i]]), axis=0)
       
        test_radiomics_1 = np.concatenate((A_features_1[idx_y1[i]], B_features_1[idx_y2[i]]), axis=0)
        test_radiomics_2 = np.concatenate((A_features_2[idx_y1[i]], B_features_2[idx_y2[i]]), axis=0)
        
        ############################################################################    
               
        label_train = np.concatenate((np.zeros((np.shape(idx_x1[i])[0], 1), dtype='uint8'),
                                      np.ones((np.shape(idx_x2[i])[0], 1), dtype='uint8')), axis=0)
        
        label_test = np.concatenate((np.zeros((np.shape(idx_y1[i])[0], 1), dtype='uint8'),
                                     np.ones((np.shape(idx_y2[i])[0], 1), dtype='uint8')), axis=0)
        
        ############################################################################
        
        train_cnn_left_1 = np.expand_dims(train_cnn_left_1, axis = 4)
        train_cnn_right_1 = np.expand_dims(train_cnn_right_1, axis = 4)
        
        train_cnn_left_2 = np.expand_dims(train_cnn_left_2, axis = 4)
        train_cnn_right_2 = np.expand_dims(train_cnn_right_2, axis = 4)

    
        test_cnn_left_1 = np.expand_dims(test_cnn_left_1, axis = 4)
        test_cnn_right_1 = np.expand_dims(test_cnn_right_1, axis = 4)

        test_cnn_left_2 = np.expand_dims(test_cnn_left_2, axis = 4)
        test_cnn_right_2 = np.expand_dims(test_cnn_right_2, axis = 4)
        
                                       
        ###########################################################################
                   # cnn + radiomics + dense classifier (classifier 1)
        ###########################################################################
        if type_classifier == 'compose-CNN-Radiomics':
            
            optimizer = opt(lr=learning_rate)
        
            if Loss_function == 'binary_crossentropy':
                last_layer_activation = 'sigmoid'
                num_classes = 1
    
            elif Loss_function == 'categorical_crossentropy':
                last_layer_activation = 'softmax'
                num_classes = 2
            
                label_train_ctg = np_utils.to_categorical(label_train)
                label_test_ctg = np_utils.to_categorical(label_test) 
            
            input_shape_hippo = train_cnn_left_1.shape[1:]
            input_shape_amyg = train_cnn_left_2.shape[1:]
            age_sex_input_shape = train_age_sex.shape[1:]
            feature_input_shape_1 = train_radiomics_1.shape[1:]
            feature_input_shape_2 = train_radiomics_2.shape[1:]
                      
            cnn_model_path = type_classifier+'_'+type_input+'_'+groups[0]+'_'+groups[1]+ '_in_fold_'+str(i+1)+'.h5'
            
        
            left_input_1 = Input(shape=input_shape_hippo, name='input1_left')
            left_model_1 = create_convolution_layers(left_input_1, side='left_1')
        
            right_input_1 = Input(shape=input_shape_hippo, name='input1_right')
            right_model_1 = create_convolution_layers(right_input_1, side='right_1')          
            
            
            left_input_2 = Input(shape=input_shape_amyg, name='input2_left')
            left_model_2 = create_convolution_layers(left_input_2, side='left_2')
        
            right_input_2 = Input(shape=input_shape_amyg, name='input2_right')
            right_model_2 = create_convolution_layers(right_input_2, side='right_2') 
  
            feature_input_age_sex = Input(shape=age_sex_input_shape, name='input_age_sex')
            
            feature_input_1 = Input(shape=feature_input_shape_1, name='input1_features')
            feature_input_2 = Input(shape=feature_input_shape_2, name='input2_features')
            
            concat1 = concatenate([left_model_1, right_model_1], name='concat1')
            flat1 = Flatten(name='flatten1')(concat1)
            
            concat2 = concatenate([left_model_2, right_model_2], name='concat2')
            flat2 = Flatten(name='flatten2')(concat2)
            
            if type_input=='compose-Hippo-Amyg':
                concat_final = concatenate([flat1, flat2, feature_input_age_sex, feature_input_1, feature_input_2], name='concat_final')
            elif type_input=='just-Hippo':
                concat_final = concatenate([flat1, feature_input_age_sex, feature_input_1], name='concat_final')
            elif type_input=='just-Amyg': 
                concat_final = concatenate([flat2, feature_input_age_sex, feature_input_2], name='concat_final')

            
            dense1 = Dense(128, kernel_initializer='glorot_normal', activation='relu', name='dense1')(concat_final)
            last_dropout = Dropout(0.4, name='last_dropout')(dense1)  
            output = Dense(num_classes, activation=last_layer_activation, name='classifier')(last_dropout)
            
            if type_input=='compose-Hippo-Amyg':
                model = Model(inputs=[left_input_1, right_input_1, left_input_2, right_input_2, feature_input_age_sex, feature_input_1, feature_input_2], outputs=[output])
            elif type_input=='just-Hippo':
                model = Model(inputs=[left_input_1, right_input_1, feature_input_age_sex, feature_input_1], outputs=[output])
            elif type_input=='just-Amyg': 
                model = Model(inputs=[left_input_2, right_input_2, feature_input_age_sex, feature_input_2], outputs=[output])
            
            model.compile(loss=Loss_function, optimizer=optimizer, metrics=['accuracy'])
         
            #model.summary()
            
            if plt_model and i == 0:
                plot_model(model, to_file='plot_model_'+ type_classifier + '_' + type_input +'.eps', show_shapes=True)
                plot_model(model, to_file='plot_model_'+ type_classifier + '_' + type_input +'.jpg', show_shapes=True)
                plot_model(model, to_file='plot_model_'+ type_classifier + '_' + type_input +'.pdf', show_shapes=True)
        
            model_checkpoint = callbacks.ModelCheckpoint(cnn_model_path,
                                                         monitor='val_loss',
                                                         save_best_only=True,
                                                         mode='min')
    
            reducelronplateau = callbacks.ReduceLROnPlateau(monitor='val_loss',
                                                            factor=0.99,
                                                            patience=5)
        
        
            callbacks_list = [model_checkpoint, reducelronplateau]
                        
            train_1, val_1, y_train_cnn, y_val_cnn = train_test_split(train_cnn_left_1, label_train, test_size=0.2, random_state=rs)
            train_2, val_2, y_train_cnn, y_val_cnn = train_test_split(train_cnn_right_1, label_train, test_size=0.2, random_state=rs)
            train_3, val_3, y_train_cnn, y_val_cnn = train_test_split(train_cnn_left_2, label_train, test_size=0.2, random_state=rs)
            train_4, val_4, y_train_cnn, y_val_cnn = train_test_split(train_cnn_right_2, label_train, test_size=0.2, random_state=rs)
            train_5, val_5, y_train_cnn, y_val_cnn = train_test_split(train_age_sex, label_train, test_size=0.2, random_state=rs)
            train_6, val_6, y_train_cnn, y_val_cnn = train_test_split(train_radiomics_1, label_train, test_size=0.2, random_state=rs)
            train_7, val_7, y_train_cnn, y_val_cnn = train_test_split(train_radiomics_2, label_train, test_size=0.2, random_state=rs)
           
            if type_input=='compose-Hippo-Amyg':
                 history = model.fit([train_1, train_2, train_3, train_4, train_5, train_6, train_7], y_train_cnn,
                                 batch_size=batch, 
                                 epochs = epochs, 
                                 callbacks = callbacks_list,
                                 validation_data = ([val_1, val_2, val_3, val_4, val_5, val_6, val_7], y_val_cnn))
        
            elif type_input=='just-Hippo':
                 history = model.fit([train_1, train_2, train_5, train_6], y_train_cnn,
                                 batch_size=batch, 
                                 epochs = epochs, 
                                 callbacks = callbacks_list,
                                 validation_data = ([val_1, val_2, val_5, val_6], y_val_cnn))
        
            elif type_input=='just-Amyg':
                 history = model.fit([train_3, train_4, train_5, train_7], y_train_cnn,
                                 batch_size=batch, 
                                 epochs = epochs, 
                                 callbacks = callbacks_list,
                                 validation_data = ([val_3, val_4, val_5, val_7], y_val_cnn))
           
            #---------------------------------------------------------------------------------------
      
            model = load_model(cnn_model_path)
            if type_input=='compose-Hippo-Amyg':
                loss1, acc1 = model.evaluate([test_cnn_left_1, test_cnn_right_1, test_cnn_left_2, test_cnn_right_2, test_age_sex, test_radiomics_1, test_radiomics_2], label_test)        
                y_pred_cnn_val = model.predict([val_1, val_2, val_3, val_4, val_5, val_6, val_7])
                y_pred_cnn_test = model.predict([test_cnn_left_1, test_cnn_right_1, test_cnn_left_2, test_cnn_right_2, test_age_sex, test_radiomics_1, test_radiomics_2])
    
            elif type_input=='just-Hippo':
                loss1, acc1 = model.evaluate([test_cnn_left_1, test_cnn_right_1, test_age_sex, test_radiomics_1], label_test)         
                y_pred_cnn_val = model.predict([val_1, val_2, val_5, val_6])
                y_pred_cnn_test = model.predict([test_cnn_left_1, test_cnn_right_1, test_age_sex, test_radiomics_1])
    
            elif type_input=='just-Amyg':
                loss1, acc1 = model.evaluate([test_cnn_left_2, test_cnn_right_2, test_age_sex, test_radiomics_2], label_test)        
                y_pred_cnn_val = model.predict([val_3, val_4, val_5, val_7])
                y_pred_cnn_test = model.predict([test_cnn_left_2, test_cnn_right_2, test_age_sex, test_radiomics_2])
    
            
            acc_kfold_classifier1.append(accuracy_score(label_test, binarize(y_pred_cnn_test, threshold=0.50, copy=True)))
            auc_kfold_classifier1.append(roc_auc_score(label_test, y_pred_cnn_test))
            
            file = open("report_(cnn_radiomics_dense)_"+type_classifier+"_"+type_input+"_fold_"+str(i+1)+".txt", "w") 
            file.write(str(classification_report(label_test, binarize(y_pred_cnn_test, threshold=0.50, copy=True))))
            file.write('\n\n\n')
            file.write('ACC : ' + str(accuracy_score(label_test, binarize(y_pred_cnn_test, threshold=0.50, copy=True))))
            file.write('\n')
            file.write('AUC : ' + str(roc_auc_score(label_test, y_pred_cnn_test)))
            file.close() 
            
            print(type_classifier +" "+type_input+" "+" fold " + str(i+1) + " accuracy_score on test set : ", str(np.float16(acc1)))        
            print("")                
        
        
        ###########################################################################
                    # cnn + radiomics + svm classifier (classifier 2)
        ###########################################################################
        if type_classifier == 'compose-CNN-Radiomics':

            activation_model = Model(inputs=model.input, outputs=model.layers[-3].output)
            # activation_model.summary()
    
            if type_input=='compose-Hippo-Amyg':
                 activations_train = activation_model.predict([train_1, train_2, train_3, train_4, train_5, train_6, train_7])
                 activations_test = activation_model.predict([test_cnn_left_1, test_cnn_right_1, test_cnn_left_2, test_cnn_right_2, test_age_sex, test_radiomics_1, test_radiomics_2])
                 activations_val = activation_model.predict([val_1, val_2, val_3, val_4, val_5, val_6, val_7])
            
            elif type_input=='just-Hippo':
                 activations_train = activation_model.predict([train_1, train_2, train_5, train_6])
                 activations_test = activation_model.predict([test_cnn_left_1, test_cnn_right_1, test_age_sex, test_radiomics_1])
                 activations_val = activation_model.predict([val_1, val_2, val_5, val_6])
            
            elif type_input=='just-Amyg':
                 activations_train = activation_model.predict([train_3, train_4, train_5, train_7])
                 activations_test = activation_model.predict([test_cnn_left_2, test_cnn_right_2, test_age_sex, test_radiomics_2])
                 activations_val = activation_model.predict([val_3, val_4, val_5, val_7])
                
                    
            y_train_svm = y_train_cnn[:,0]
            y_test_svm = label_test[:,0]
            y_val_svm = y_val_cnn[:,0]
            
            clf = GridSearchCV(SVC(class_weight=None, decision_function_shape='ovr',
                               gamma='scale', probability=True,
                               shrinking=True, tol=0.001, verbose=False),
                           svm_params, cv=3, n_jobs=-1)
            
            clf.fit(activations_train, y_train_svm)  
          
            y_pred = clf.predict(activations_test) 
            y_pred_svm_val = clf.predict_proba(activations_val)
            y_pred_svm_test = clf.predict_proba(activations_test)
            
            acc2 = accuracy_score(y_test_svm, y_pred) 
      
            acc_kfold_classifier2.append(accuracy_score(y_test_svm, y_pred))
            auc_kfold_classifier2.append(roc_auc_score(y_test_svm, y_pred_svm_test[:, 1]))
            
            file = open("report_(cnn_radiomics_svm)_"+type_classifier+"_"+type_input+"_fold_"+str(i+1)+".txt", "w") 
            file.write(str(classification_report(y_test_svm, y_pred)))
            file.write('\n\n\n')
            file.write('ACC : ' + str(accuracy_score(y_test_svm, y_pred)))
            file.write('\n')
            file.write('AUC : ' + str(roc_auc_score(y_test_svm, y_pred_svm_test[:, 1])))
            file.close() 
            
            print(type_classifier +" "+type_input+" "+" fold " + str(i+1) + " accuracy_score on test set : ", str(np.float16(acc2)))        
            print("")           
            
        
        ###########################################################################
                    # radiomics + svm classifier (classifier 3)
        ###########################################################################
        if type_classifier == 'compose-CNN-Radiomics' or type_classifier == 'just-Radiomics':
        
            if type_classifier == 'just-Radiomics':
                train_1, val_1, y_train_cnn, y_val_cnn = train_test_split(train_cnn_left_1, label_train, test_size=0.2, random_state=rs)
                train_2, val_2, y_train_cnn, y_val_cnn = train_test_split(train_cnn_right_1, label_train, test_size=0.2, random_state=rs)
                train_3, val_3, y_train_cnn, y_val_cnn = train_test_split(train_cnn_left_2, label_train, test_size=0.2, random_state=rs)
                train_4, val_4, y_train_cnn, y_val_cnn = train_test_split(train_cnn_right_2, label_train, test_size=0.2, random_state=rs)
                train_5, val_5, y_train_cnn, y_val_cnn = train_test_split(train_age_sex, label_train, test_size=0.2, random_state=rs)
                train_6, val_6, y_train_cnn, y_val_cnn = train_test_split(train_radiomics_1, label_train, test_size=0.2, random_state=rs)
                train_7, val_7, y_train_cnn, y_val_cnn = train_test_split(train_radiomics_2, label_train, test_size=0.2, random_state=rs)
         
            y_train_svm = y_train_cnn[:,0]
            y_test_svm = label_test[:,0]
            y_val_svm = y_val_cnn[:,0]
            
            clf = GridSearchCV(SVC(class_weight=None, decision_function_shape='ovr',
                               gamma='scale', probability=True,
                               shrinking=True, tol=0.001, verbose=False),
                           svm_params, cv=3, n_jobs=-1)
            
            
            if type_input=='compose-Hippo-Amyg':
                clf.fit(np.concatenate((train_6, train_7), axis=1), y_train_svm)     
                y_pred = clf.predict(np.concatenate((test_radiomics_1, test_radiomics_2), axis=1))     
                y_pred_svm_radiomics_val = clf.predict_proba(np.concatenate((train_6, train_7), axis=1))
                y_pred_svm_radiomics_test = clf.predict_proba(np.concatenate((test_radiomics_1, test_radiomics_2), axis=1))
            
            elif type_input=='just-Hippo':
                clf.fit(train_6, y_train_svm)      
                y_pred = clf.predict(test_radiomics_1)     
                y_pred_svm_radiomics_val = clf.predict_proba(train_6)
                y_pred_svm_radiomics_test = clf.predict_proba(test_radiomics_1)
            
            elif type_input=='just-Amyg':
                clf.fit(train_7, y_train_svm)
                y_pred = clf.predict(test_radiomics_2)     
                y_pred_svm_radiomics_val = clf.predict_proba(train_7)
                y_pred_svm_radiomics_test = clf.predict_proba(test_radiomics_2)
            
            
            acc3 = accuracy_score(y_test_svm, y_pred)  
            
            acc_kfold_classifier3.append(accuracy_score(y_test_svm, y_pred))
            auc_kfold_classifier3.append(roc_auc_score(y_test_svm, y_pred_svm_radiomics_test[:, 1]))
            
            file = open("report_(radiomics_svm)_"+type_classifier+"_"+type_input+"_fold_"+str(i+1)+".txt", "w") 
            file.write(str(classification_report(y_test_svm, y_pred)))
            file.write('\n\n\n')
            file.write('ACC : ' + str(accuracy_score(y_test_svm, y_pred)))
            file.write('\n')
            file.write('AUC : ' + str(roc_auc_score(y_test_svm, y_pred_svm_radiomics_test[:, 1])))
            file.close() 
            
            print(type_classifier +" "+type_input+" "+" fold " + str(i+1) + " accuracy_score on test set : ", str(np.float16(acc3)))        
            print("")  
       
        #######################################################################
                                 # Majority Voting
        #######################################################################

        if type_classifier == 'compose-CNN-Radiomics':
            
            X1_C = np.round(y_pred_cnn_test)
            X2_C = np.expand_dims((np.round(y_pred_svm_test[:,1])), axis=1)
            X3_C = np.expand_dims((np.round(y_pred_svm_radiomics_test[:,1])), axis=1)
    
#            for x in range(np.shape(y_pred_cnn_test)[0]):
#                if y_pred_cnn_test[x] < 0.5:
#                    y_pred_cnn_test[x] = 1 - y_pred_cnn_test[x]
            
            X1_P = y_pred_cnn_test
            X2_P = np.expand_dims(y_pred_svm_test[:, 1], axis=1) 
            #X2_P = np.expand_dims(np.max(y_pred_svm_test, axis=1), axis=1) 
            X3_P = np.expand_dims(y_pred_svm_radiomics_test[:, 1], axis=1)
            #X3_P = np.expand_dims(np.max(y_pred_svm_radiomics_test, axis=1), axis=1)

          
            probabilities = np.concatenate((X1_P, X2_P, X3_P), axis=1)
            categories = np.concatenate((X1_C, X2_C, X3_C), axis=1)
                   
            voting = np.sum(categories,  axis=1)
            voting[np.where(voting < 2)] = 0
            voting[np.where(voting >= 2)] = 1
            
            pbs = np.divide(np.sum(probabilities, axis=1), 3)
            
            acc_mv = accuracy_score(y_test_svm, voting) 
            
            acc_kfold_classifier_mv.append(accuracy_score(y_test_svm, voting))
            auc_kfold_classifier_mv.append(roc_auc_score(y_test_svm, pbs))
            
            
            file = open("report_MV_"+type_classifier+"_"+type_input+"_fold_"+str(i+1)+".txt", "w") 
            file.write(str(classification_report(y_test_svm, voting)))
            file.write('\n\n\n')
            file.write('ACC : ' + str(accuracy_score(y_test_svm, voting)))
            file.write('\n')
            file.write('AUC : ' + str(roc_auc_score(y_test_svm, pbs)))
            file.close() 
            
            print("MV "+type_classifier +" "+type_input+" "+" fold " + str(i+1) + " accuracy_score on test set : ", str(np.float16(acc_mv)))        
            print("") 
          
        #######################################################################
         # cnn + dense classifier (classifier 4 - not use in majority voting)
        #######################################################################
        if type_classifier == 'just-CNN':
            
            optimizer = opt(lr=learning_rate)
        
            if Loss_function == 'binary_crossentropy':
                last_layer_activation = 'sigmoid'
                num_classes = 1
    
            elif Loss_function == 'categorical_crossentropy':
                last_layer_activation = 'softmax'
                num_classes = 2
            
                label_train_ctg = np_utils.to_categorical(label_train)
                label_test_ctg = np_utils.to_categorical(label_test) 
            
            
            input_shape_hippo = train_cnn_left_1.shape[1:]
            input_shape_amyg = train_cnn_left_2.shape[1:]
            age_sex_input_shape = train_age_sex.shape[1:]
                      
            cnn_model_path = type_classifier+'_'+type_input+'_'+groups[0]+'_'+groups[1]+ '_in_fold_'+str(i+1)+'.h5'
        
        
            left_input_1 = Input(shape=input_shape_hippo, name='input1_left')
            left_model_1 = create_convolution_layers(left_input_1, side='left_1')
        
            right_input_1 = Input(shape=input_shape_hippo, name='input1_right')
            right_model_1 = create_convolution_layers(right_input_1, side='right_1')          
            
            
            left_input_2 = Input(shape=input_shape_amyg, name='input2_left')
            left_model_2 = create_convolution_layers(left_input_2, side='left_2')
        
            right_input_2 = Input(shape=input_shape_amyg, name='input2_right')
            right_model_2 = create_convolution_layers(right_input_2, side='right_2') 
  
            feature_input_age_sex = Input(shape=age_sex_input_shape, name='input_age_sex')
                        
            concat1 = concatenate([left_model_1, right_model_1], name='concat1')
            flat1 = Flatten(name='flatten1')(concat1)
            
            concat2 = concatenate([left_model_2, right_model_2], name='concat2')
            flat2 = Flatten(name='flatten2')(concat2)
            
            if type_input=='compose-Hippo-Amyg':
                concat_final = concatenate([flat1, flat2, feature_input_age_sex], name='concat_final')
            elif type_input=='just-Hippo':
                concat_final = concatenate([flat1, feature_input_age_sex], name='concat_final')
            elif type_input=='just-Amyg': 
                concat_final = concatenate([flat2, feature_input_age_sex], name='concat_final')

            
            dense1 = Dense(128, kernel_initializer='glorot_normal', activation='relu', name='dense1')(concat_final)
            last_dropout = Dropout(0.4, name='last_dropout')(dense1)  
            output = Dense(num_classes, activation=last_layer_activation, name='classifier')(last_dropout)
            
            if type_input=='compose-Hippo-Amyg':
                model = Model(inputs=[left_input_1, right_input_1, left_input_2, right_input_2, feature_input_age_sex], outputs=[output])
            elif type_input=='just-Hippo':
                model = Model(inputs=[left_input_1, right_input_1, feature_input_age_sex], outputs=[output])
            elif type_input=='just-Amyg': 
                model = Model(inputs=[left_input_2, right_input_2, feature_input_age_sex], outputs=[output])
            
            model.compile(loss=Loss_function, optimizer=optimizer, metrics=['accuracy'])
         
            #model.summary()
            
            if plt_model and i == 0:
                plot_model(model, to_file='plot_model_'+ type_classifier + '_' + type_input +'.eps', show_shapes=True)
                plot_model(model, to_file='plot_model_'+ type_classifier + '_' + type_input +'.jpg', show_shapes=True)
                plot_model(model, to_file='plot_model_'+ type_classifier + '_' + type_input +'.pdf', show_shapes=True)
        
            model_checkpoint = callbacks.ModelCheckpoint(cnn_model_path,
                                                         monitor='val_loss',
                                                         save_best_only=True,
                                                         mode='min')
    
            reducelronplateau = callbacks.ReduceLROnPlateau(monitor='val_loss',
                                                            factor=0.99,
                                                            patience=5)
        
        
            callbacks_list = [model_checkpoint, reducelronplateau]
                        
            train_1, val_1, y_train_cnn, y_val_cnn = train_test_split(train_cnn_left_1, label_train, test_size=0.2, random_state=rs)
            train_2, val_2, y_train_cnn, y_val_cnn = train_test_split(train_cnn_right_1, label_train, test_size=0.2, random_state=rs)
            train_3, val_3, y_train_cnn, y_val_cnn = train_test_split(train_cnn_left_2, label_train, test_size=0.2, random_state=rs)
            train_4, val_4, y_train_cnn, y_val_cnn = train_test_split(train_cnn_right_2, label_train, test_size=0.2, random_state=rs)
            train_5, val_5, y_train_cnn, y_val_cnn = train_test_split(train_age_sex, label_train, test_size=0.2, random_state=rs)
           
            if type_input=='compose-Hippo-Amyg':
                 history = model.fit([train_1, train_2, train_3, train_4, train_5], y_train_cnn,
                                 batch_size=batch, 
                                 epochs = epochs, 
                                 callbacks = callbacks_list,
                                 validation_data = ([val_1, val_2, val_3, val_4, val_5], y_val_cnn))
        
            elif type_input=='just-Hippo':
                 history = model.fit([train_1, train_2, train_5], y_train_cnn,
                                 batch_size=batch, 
                                 epochs = epochs, 
                                 callbacks = callbacks_list,
                                 validation_data = ([val_1, val_2, val_5], y_val_cnn))
        
            elif type_input=='just-Amyg':
                 history = model.fit([train_3, train_4, train_5], y_train_cnn,
                                 batch_size=batch, 
                                 epochs = epochs, 
                                 callbacks = callbacks_list,
                                 validation_data = ([val_3, val_4, val_5], y_val_cnn))
           
            #---------------------------------------------------------------------------------------
      
            model = load_model(cnn_model_path)
            if type_input=='compose-Hippo-Amyg':
                loss4, acc4 = model.evaluate([test_cnn_left_1, test_cnn_right_1, test_cnn_left_2, test_cnn_right_2, test_age_sex], label_test)        
                y_pred_cnn_val = model.predict([val_1, val_2, val_3, val_4, val_5])
                y_pred_cnn_test = model.predict([test_cnn_left_1, test_cnn_right_1, test_cnn_left_2, test_cnn_right_2, test_age_sex])
    
            elif type_input=='just-Hippo':
                loss4, acc4 = model.evaluate([test_cnn_left_1, test_cnn_right_1, test_age_sex], label_test)         
                y_pred_cnn_val = model.predict([val_1, val_2, val_5])
                y_pred_cnn_test = model.predict([test_cnn_left_1, test_cnn_right_1, test_age_sex])
    
            elif type_input=='just-Amyg':
                loss4, acc4 = model.evaluate([test_cnn_left_2, test_cnn_right_2, test_age_sex], label_test)        
                y_pred_cnn_val = model.predict([val_3, val_4, val_5])
                y_pred_cnn_test = model.predict([test_cnn_left_2, test_cnn_right_2, test_age_sex])
    
            
            acc_kfold_classifier4.append(accuracy_score(label_test, binarize(y_pred_cnn_test, threshold=0.50, copy=True)))
            auc_kfold_classifier4.append(roc_auc_score(label_test, y_pred_cnn_test))
            
            file = open("report_(cnn_dense)_"+type_classifier+"_"+type_input+"_fold_"+str(i+1)+".txt", "w") 
            file.write(str(classification_report(label_test, binarize(y_pred_cnn_test, threshold=0.50, copy=True))))
            file.write('\n\n\n')
            file.write('ACC : ' + str(accuracy_score(label_test, binarize(y_pred_cnn_test, threshold=0.50, copy=True))))
            file.write('\n')
            file.write('AUC : ' + str(roc_auc_score(label_test, y_pred_cnn_test)))
            file.close() 
            
            print(type_classifier +" "+type_input+" "+" fold " + str(i+1) + " accuracy_score on test set : ", str(np.float16(acc4)))        
            print("")    
        
        #######################################################################
                                 # 
        #######################################################################
    
    if type_classifier == 'just-CNN':
        
        file = open("report_(mean_kfold)_"+type_classifier+"_"+type_input+".txt", "w") 
        file.write("mean of acc on test set: " + str(np.mean(acc_kfold_classifier4)))
        file.write('\n\n\n')
        file.write("mean of auc on test set: " + str(np.mean(auc_kfold_classifier4)))
        file.write('\n')
        file.close() 
                     
        print("Mean of acc on test set : " + str(np.mean(acc_kfold_classifier4)))   
        print("Mean of auc on test set : " + str(np.mean(auc_kfold_classifier4)))

    elif type_classifier == 'just-Radiomics':
        
        file = open("report_(mean_kfold)_"+type_classifier+"_"+type_input+".txt", "w") 
        file.write("mean of acc on test set: " + str(np.mean(acc_kfold_classifier3)))
        file.write('\n\n\n')
        file.write("mean of auc on test set: " + str(np.mean(auc_kfold_classifier3)))
        file.write('\n')
        file.close() 
                     
        print("Mean of acc on test set : " + str(np.mean(acc_kfold_classifier3)))   
        print("Mean of auc on test set : " + str(np.mean(auc_kfold_classifier3)))
 
        
    elif type_classifier == 'compose-CNN-Radiomics':
        
        file = open("report_(mean_kfold)_"+type_classifier+"_"+type_input+".txt", "w") 
        file.write("mean of acc on test set 1 : " + str(np.mean(acc_kfold_classifier1)))
        file.write('\n')
        file.write("mean of acc on test set 2 : " + str(np.mean(acc_kfold_classifier2)))
        file.write('\n')
        file.write("mean of acc on test set 3 : " + str(np.mean(acc_kfold_classifier3)))
        file.write('\n')
        file.write("mean of acc on test set mv: " + str(np.mean(acc_kfold_classifier_mv)))
        file.write('\n\n\n')
        file.write("mean of auc on test set 1: " + str(np.mean(auc_kfold_classifier1)))
        file.write('\n')
        file.write("mean of auc on test set 2: " + str(np.mean(auc_kfold_classifier2)))
        file.write('\n')
        file.write("mean of auc on test set 3: " + str(np.mean(auc_kfold_classifier3)))
        file.write('\n')
        file.write("mean of auc on test set mv: " + str(np.mean(auc_kfold_classifier_mv)))
        file.write('\n')
        file.close() 
                 
        print("mean of acc on test set : " + str(np.mean(acc_kfold_classifier1)))
        print("mean of acc on test set : " + str(np.mean(acc_kfold_classifier2)))
        print("mean of acc on test set : " + str(np.mean(acc_kfold_classifier3)))
    
        print("Mean of Majority Voting acc on test set : " + str(np.mean(acc_kfold_classifier_mv)))
    
        print("mean of auc on test set : " + str(np.mean(auc_kfold_classifier1)))
        print("mean of auc on test set : " + str(np.mean(auc_kfold_classifier2)))
        print("mean of auc on test set : " + str(np.mean(auc_kfold_classifier3)))
    
        print("Mean of Majority Voting auc on test set : " + str(np.mean(auc_kfold_classifier_mv)))
