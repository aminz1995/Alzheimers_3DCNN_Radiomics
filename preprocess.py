#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Amin Zarei (Gmail: aminz1995@gmail.com)
"""
#------------------------------------------------------------------------------
# import global modules
import os
import numpy as np
import nipype
import SimpleITK as sitk
import scipy
import gzip
import shutil
# import local modules
import visualization as vs

#------------------------------------------------------------------------------
# config directories

AD_path = './ADNI/T1-weighted/MPR__GradWarp__B1_Correction__N3__Scaled/AD/'
CN_path = './ADNI/T1-weighted/MPR__GradWarp__B1_Correction__N3__Scaled/CN/'
MCI_path = './ADNI/T1-weighted/MPR__GradWarp__B1_Correction__N3__Scaled/MCI/'

Brain_AD_path = './ADNI/BrainExtracted/FreeSurfer_Cross-Sectional_Processing_brainmask/AD_FS_B_mask_CS/'
Brain_CN_path = './ADNI/BrainExtracted/FreeSurfer_Cross-Sectional_Processing_brainmask/CN_FS_B_mask_CS/'
Brain_MCI_path = './ADNI/BrainExtracted/FreeSurfer_Cross-Sectional_Processing_brainmask/MCI_FS_B_mask_CS/'
#------------------------------------------------------------------------------
# Define Functions : ##########################################################
  
def load_names_1(root_path):
    
    main_images = []
    masks = []
    # extract features from AD images 
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
                            masks.append(filepath)
                            
                        if "GradWarp" in filepath and folder in filepath:
                            main_images.append(filepath)

    return main_images, masks   
 
###############################################################################
    
def load_names_2(root_path):
    
    brain_images = []
    hippo_masks = []
    # extract features from AD images 
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
                            
                        if "brainmask" in filepath and folder in filepath: # or Brain_Mask
                            brain_images.append(filepath)

    return brain_images, hippo_masks   
 
###############################################################################
    
def load_names_3(root_path):
    
    brain_mask_names = []
    
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
                            brain_mask_names.append(filepath)

    return brain_mask_names  
 
###############################################################################
    
def skull_stripping(data_dir, frac):
    
    for address in data_dir:
        try:
            mybet = nipype.interfaces.fsl.BET(in_file=os.path.join(address),out_file=os.path.join(address+"Brain_Mask.nii"), frac=frac)                #frac=0.2
            mybet.run()                                                                                                                                      #executing the brain extraction
            print(address+' is skull stripped')
        except:
            print(address+' is not skull stripped')

###############################################################################
            
def decompress_brain_mask(path):

     for subdir, dirs, files in os.walk(path):
                for file in files:
                    filepath = subdir + os.sep + file
                    if filepath.endswith(".gz"): 
                        
                        with gzip.open(filepath, 'rb') as f_in:
                            with open(filepath[:-3], 'wb') as f_out:
                                shutil.copyfileobj(f_in, f_out)  
                        
            
###############################################################################

def generate_LR_hippo(brain, hippo):
    
     for i in range(np.shape(brain)[0]):
        
        print(str(i+1) + " of " + str(np.shape(brain)[0]))
        
        Image = sitk.ReadImage(brain[i])
        Mask = sitk.ReadImage(hippo[i])
    
        image = sitk.GetArrayFromImage(Image)   
        mask = sitk.GetArrayFromImage(Mask)
    
        image = np.rot90(image, axes=(0,2))
        image = np.rot90(image, axes=(0,1))
    
        mask = np.rot90(mask, axes=(0,1))
        mask = np.flip(mask, axis=0)      
        
        mask[np.where(mask!=0)] = 1 
    
        merge = image * mask
        
        left = merge[:, :, 0:int(np.shape(merge)[2]/2)]
        right = merge[:, :, int(np.shape(merge)[2]/2):]
        
        coords = np.argwhere(left>0)
        x1,y1,z1 = coords.min(axis=0)
        x2,y2,z2 = coords.max(axis=0) + 1
        left_hippo = left[x1:x2, y1:y2, z1:z2]
        
        coords = np.argwhere(right>0)
        x1,y1,z1 = coords.min(axis=0)
        x2,y2,z2 = coords.max(axis=0) + 1
        right_hippo = right[x1:x2, y1:y2, z1:z2]
                
#        dic = {'test':list(merge)}
#        scipy.io.savemat("test", dic, appendmat=True, format='5', long_field_names=False, do_compression=False, oned_as='row') 
       
        dic = {'left_hippo':list(left_hippo)}
        scipy.io.savemat(hippo[i][:-4]+"_left_hippo.mat", dic, appendmat=True, format='5', long_field_names=False, do_compression=False, oned_as='row') 

        dic = {'right_hippo':list(right_hippo)}
        scipy.io.savemat(hippo[i][:-4]+"_right_hippo.mat", dic, appendmat=True, format='5', long_field_names=False, do_compression=False, oned_as='row') 

###############################################################################

def generate_LR_hippo_patch(brain, hippo):
    
     for i in range(np.shape(brain)[0]):
        
        print(str(i+1) + " of " + str(np.shape(brain)[0]))
        
        Image = sitk.ReadImage(brain[i])
        Mask = sitk.ReadImage(hippo[i])
    
        image = sitk.GetArrayFromImage(Image)   
        mask = sitk.GetArrayFromImage(Mask)
    
        image = np.rot90(image, axes=(0,2))
        image = np.rot90(image, axes=(0,1))
    
        mask = np.rot90(mask, axes=(0,1))
        mask = np.flip(mask, axis=0)      
        
        mask[np.where(mask!=0)] = 1 
    
        merge = image * mask
        
        left = merge[:, :, 0:int(np.shape(merge)[2]/2)]
        right = merge[:, :, int(np.shape(merge)[2]/2):]
        
        left_im = image[:, :, 0:int(np.shape(image)[2]/2)]
        right_im = image[:, :, int(np.shape(image)[2]/2):]
        
        coords = np.argwhere(left>0)
        x1,y1,z1 = coords.min(axis=0)
        x2,y2,z2 = coords.max(axis=0) + 1
        left_hippo = left_im[x1:x2, y1:y2, z1:z2]
        
        coords = np.argwhere(right>0)
        x1,y1,z1 = coords.min(axis=0)
        x2,y2,z2 = coords.max(axis=0) + 1
        right_hippo = right_im[x1:x2, y1:y2, z1:z2]
                
#        dic = {'test':list(merge)}
#        scipy.io.savemat("test", dic, appendmat=True, format='5', long_field_names=False, do_compression=False, oned_as='row') 
       
        dic = {'left_hippo':list(left_hippo)}
        scipy.io.savemat(hippo[i][:-4]+"_left_hippo_patch.mat", dic, appendmat=True, format='5', long_field_names=False, do_compression=False, oned_as='row') 

        dic = {'right_hippo':list(right_hippo)}
        scipy.io.savemat(hippo[i][:-4]+"_right_hippo_patch.mat", dic, appendmat=True, format='5', long_field_names=False, do_compression=False, oned_as='row') 

###############################################################################           
    
def find_brain_mask_match_with_hippo(Brains, hippos):
    
    for Brain in Brains:
        for hippo in hippos:
            x1 = hippo.split('/')[-3]
            x2 = hippo.split('/')[-4]
            
            y1 = Brain.split('/')[-3]
            y2 = Brain.split('/')[-5]
            
            if x1 == y1 and x2 == y2:
                  
                f_in = Brain
                f_out = hippo.replace(hippo.split('/')[-1], Brain.split('/')[-1])
                shutil.copy(f_in, f_out)  
    
       
###############################################################################

        
if __name__ == "__main__":
    
    #--------------------------------------------------------------------------
    
    AD_main_images, AD_masks = load_names_1(AD_path)
    skull_stripping(AD_main_images, frac=0.5)
    decompress_brain_mask(AD_path)
    
    AD_brain_images, AD_hippo = load_names_2(AD_path)    
    generate_LR_hippo(AD_brain_images, AD_hippo)
    generate_LR_hippo_patch(AD_brain_images, AD_hippo) 

    
    Brain_AD_names = load_names_3(Brain_AD_path)
    find_brain_mask_match_with_hippo(Brain_AD_names, AD_hippo)
    find_ventricle(AD_main_images)
    
    #---------------------------------------------------------------------------
    
    CN_main_images, CN_masks = load_names_1(CN_path)
    skull_stripping(CN_main_images, frac=0.5)
    decompress_brain_mask(CN_path)
    
    CN_brain_images, CN_hippo = load_names_2(CN_path)
    generate_LR_hippo(CN_brain_images, CN_hippo)    
    generate_LR_hippo_patch(CN_brain_images, CN_hippo)    

    Brain_CN_names = load_names_3(Brain_CN_path)
    find_brain_mask_match_with_hippo(Brain_CN_names, CN_hippo)
    find_ventricle(CN_main_images)
    
    #---------------------------------------------------------------------------
    
    MCI_main_images, MCI_masks = load_names_1(MCI_path)
    skull_stripping(MCI_main_images, frac=0.5)
    decompress_brain_mask(MCI_path)
   
    MCI_brain_images, MCI_hippo = load_names_2(MCI_path)
    generate_LR_hippo(MCI_brain_images, MCI_hippo)  
    generate_LR_hippo_patch(MCI_brain_images, MCI_hippo)    


    Brain_MCI_names = load_names_3(Brain_MCI_path)
    find_brain_mask_match_with_hippo(Brain_MCI_names, MCI_hippo)
    find_ventricle(MCI_main_images)
    
   #---------------------------------------------------------------------------
            
    Image = sitk.ReadImage(AD_brain_images[1])
    image = sitk.GetArrayFromImage(Image) 
    vs.multi_slice_viewer(image)
