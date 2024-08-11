#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Amin Zarei (Gmail: aminz1995@gmail.com)
"""
#------------------------------------------------------------------------------

import os
import shutil
import gzip
import numpy as np
import scipy.io as sio
import SimpleITK as sitk

path = './ADNI/BrainExtracted/FreeSurfer_Cross-Sectional_Processing_brainmask/'

###############################################################################
            
def decompress_gz(path, structures):

     for subdir, dirs, files in os.walk(path):
                for file in files:
                    filepath = subdir + os.sep + file
                    if filepath.endswith("brainmask-"+structures[0]+"_corr.nii.gz") or filepath.endswith("brainmask-"+structures[1]+"_corr.nii.gz"): 
                        
                        with gzip.open(filepath, 'rb') as f_in:
                            with open(filepath[:-3], 'wb') as f_out:
                                shutil.copyfileobj(f_in, f_out)  
                        
            
###############################################################################
            
def delete_nii(path, structures):

     for subdir, dirs, files in os.walk(path):
                for file in files:
                    filepath = subdir + os.sep + file
                    if filepath.endswith("brainmask-"+structures[0]+"_corr.nii") or filepath.endswith("brainmask-"+structures[1]+"_corr.nii"): 
                        os.remove(filepath)
                        
            
###############################################################################

def prepare_ROI_for_CNN(root_path, roi):
    
    total_subjects = len(os.listdir(root_path))
    print('total subjects:', total_subjects)
    subjects = os.listdir(root_path)
    
    for i in range(np.shape(subjects)[0]):
        
        print(str(i+1) + ' of ' + str(np.shape(subjects)[0]))
        
        subject_images_dir = root_path + subjects[i] + "/FreeSurfer_Cross-Sectional_Processing_brainmask/"
        subject_images = os.listdir(subject_images_dir)
        for folder in subject_images:
            path = subject_images_dir + folder + os.sep
            
            for subdir, dirs, files in os.walk(path):
                for file in files:
                    filepath = subdir + os.sep + file
                    
                    if filepath.endswith("brainmask.nii") and folder in filepath:
                        brain_name = filepath
                        mask_name = brain_name.split('.nii')[0] + '-' + roi + "_corr.nii.gz"
                        
                        Image = sitk.ReadImage(brain_name)
                        Mask = sitk.ReadImage(mask_name)
    
                        image = sitk.GetArrayFromImage(Image)   
                        mask = sitk.GetArrayFromImage(Mask)
        
                        mask[np.where(mask!=0)] = 1 
    
                        merge = image * mask
        
                        coords = np.argwhere(merge>0)
                        x1,y1,z1 = coords.min(axis=0)
                        x2,y2,z2 = coords.max(axis=0) + 1
                        ROI = merge[x1:x2, y1:y2, z1:z2]
      
                        dic = {roi:list(ROI)}
                        sio.savemat(mask_name[:-7]+ "_Cropped.mat", dic, appendmat=True, format='5', 
                                    long_field_names=True, do_compression=False, oned_as='row') 

###############################################################################
                       

# decompress_gz(path, structures)
# print('decompressing done...')



diagnosis = ['AD','CN','MCI']
ROIs = ['L_Amyg','L_Hipp','R_Amyg','R_Hipp']

for diag in diagnosis:
    
    root_path = path + diag + '_FS_B_mask_CS/ADNI/'
    
    for roi in ROIs:
        
        prepare_ROI_for_CNN(root_path, roi)
        print('data preparation (cropping) done for ' + roi + ' structure in ' + diag + ' diagnosis group...')



# use this after feature extraction -------------------------------------------
        
# delete_nii(path, structures)
# print('nii files removed...')






