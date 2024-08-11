#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Amin Zarei (Gmail: aminz1995@gmail.com)
"""
#------------------------------------------------------------------------------

import os
from scipy.ndimage import zoom
from radiomics import featureextractor, getFeatureClasses
import numpy as np
import SimpleITK as sitk
import pickle
import scipy

###############################################################################

# config directories
AD_path = './ADNI/T1-weighted/MPR__GradWarp__B1_Correction__N3__Scaled/AD/'
CN_path = './ADNI/T1-weighted/MPR__GradWarp__B1_Correction__N3__Scaled/CN/'
MCI_path = './ADNI/T1-weighted/MPR__GradWarp__B1_Correction__N3__Scaled/MCI/'
param_path = "./"

###############################################################################

def get_feature_vector(imageName, maskName, param_path):

    paramsFile = os.path.abspath(os.path.join(param_path, 'radiomic_params.yaml'))

    if imageName is None or maskName is None:  
        print('Error getting testcase!')
        exit()

    extractor = featureextractor.RadiomicsFeatureExtractor(paramsFile)
    featureClasses = getFeatureClasses()

    print("Calculating features")
    featureVector = extractor.execute(imageName, maskName)

   for featureName in featureVector.keys():
       print("Computed %s: %s" % (featureName, featureVector[featureName]))

    return featureVector, featureClasses

###############################################################################
   
def load_names(root_path):
    
    brain_images = []
    hippo_masks = []

    total_subjects = len(os.listdir(root_path))
    print('total subjects:', total_subjects)
    subjects = os.listdir(root_path)
    
    for i in range(np.shape(subjects)[0]):
        subject_images_dir = root_path + subjects[i] + "/"  # write for loop for this
        subject_images = os.listdir(subject_images_dir)
        for folder in subject_images:
            path = subject_images_dir + folder + "/"
            
            for subdir, dirs, files in os.walk(path):
                for file in files:
                    filepath = subdir + os.sep + file
                    if filepath.endswith(".nii"):                       
                        if "Hippocampal_Mask" in filepath and folder in filepath:
                            hippo_masks.append(filepath)
                            
                        if "Grad" in filepath and "Brain_Mask" not in filepath and "brainmask" not in filepath and folder in filepath: # or Brain_Mask
                            brain_images.append(filepath)

    return brain_images, hippo_masks   
 
###############################################################################

def run(brain_names, mask_names, param_path, side):
    
    features = []
    for i in range(np.shape(brain_names)[0]):
        
        print(str(i+1) + " of " + str(np.shape(brain_names)[0]))
        
        Image = sitk.ReadImage(brain_names[i])
        Mask = sitk.ReadImage(mask_names[i])
    
        image = sitk.GetArrayFromImage(Image)   
        mask = sitk.GetArrayFromImage(Mask)
 
        image = np.rot90(image, axes=(0,2))
        image = np.rot90(image, axes=(0,1))
    
        mask = np.rot90(mask, axes=(0,1))
        mask = np.flip(mask, axis=0)
        
#        image = zoom(image,(200/image.shape[0], 200/image.shape[1], 200/image.shape[2]))
#        mask = zoom(mask,(200/mask.shape[0], 200/mask.shape[1], 200/mask.shape[2]))
#               
        mask[np.where(mask!=0)] = 1 
        l = mask.copy()
        r = mask.copy()
    
        l[:, :, 0:int(np.shape(mask)[2]/2)] = 0
        r[:, :, int(np.shape(mask)[2]/2):] = 0
    
        left = r
        right = l
        
        if side == 'both':
            Image = sitk.GetImageFromArray(image)
            Mask = sitk.GetImageFromArray(mask)
            
        if side == 'right':
            Image = sitk.GetImageFromArray(image)
            Mask = sitk.GetImageFromArray(right)
            
        if side == 'left':
            Image = sitk.GetImageFromArray(image)
            Mask = sitk.GetImageFromArray(left)
                   

        featureVector, featureClasses = get_feature_vector(Image, Mask, param_path)
        features.append(featureVector)

    return features, featureClasses

###############################################################################

def save_as_pkl(data, name):
    
    dict = data
    f = open(name + ".pkl", "wb")
    pickle.dump(dict, f)
    f.close()
    
###############################################################################
      
if __name__ == "__main__":
    
    AD_brain_images, AD_masks = load_names(AD_path)  
    CN_brain_images, CN_masks = load_names(CN_path)
    MCI_brain_images, MCI_masks = load_names(MCI_path)
        
    ###########################################################################
    ############################ feature extraction ###########################
    ###########################################################################
    Side = 'left' 
    
    AD_features, featureClasses = run(AD_brain_images, AD_masks, param_path, side=Side)
    save_as_pkl(AD_features, "AD_features")
        
    CN_features, featureClasses = run(CN_brain_images, CN_masks, param_path, side=Side)    
    save_as_pkl(CN_features, "CN_features")
    
    MCI_features, featureClasses = run(MCI_brain_images, MCI_masks, param_path, side=Side)
    save_as_pkl(MCI_features, "MCI_features")
        
    scipy.io.savemat("featureClasses", featureClasses, appendmat=True, format='5', long_field_names=True, do_compression=False, oned_as='row')  

    
    

