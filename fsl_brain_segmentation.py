#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Amin Zarei (Gmail: aminz1995@gmail.com)
"""
#------------------------------------------------------------------------------

import os
import numpy as np
import scipy.io as sio
from datetime import datetime

#------------------------------------------------------------------------------
path = './ADNI/BrainExtracted/FreeSurfer_Cross-Sectional_Processing_brainmask/'
#------------------------------------------------------------------------------
def address_generator(path):
    addresses = []
    for subdir, dirs, files in os.walk(path):
        for file in files:
            filepath = subdir + os.sep + file
            if filepath.endswith("brainmask.nii"):
                addresses.append(filepath)               
    return addresses
#------------------------------------------------------------------------------
# Structures
    
# Putamen ------------------------> (L_Puta, R_Puta)
# Caudate nucleus ----------------> (L_Caud, R_Caud)
# Nucleus accumbens --------------> (L_Accu, R_Accu)
# Globus pallidus ----------------> (L_Pall, R_Pall)
# Hippocampus --------------------> (L_Hipp, R__Hipp)
# Amygdala -----------------------> (L_Amyg, R_Amyg)
# Thalamus -----------------------> (L_Thal, R_Thal)
    
# Brainstem ----------------------> (BrStem) 

    
all_addresses = address_generator(path)

#preccessed_addresses = list(sio.loadmat('preccessed_addresses.mat')['preccessed_addresses'])
preccessed_addresses = []  # just for first run
no_proccessed_addresses = []

for add in all_addresses:
    if add not in preccessed_addresses:
        no_proccessed_addresses.append(add)  

for n, address in enumerate(no_proccessed_addresses):
    
    Input = address
    Output = address
    
    # L_Accu L_Amyg L_Caud L_Hipp L_Pall L_Puta L_Thal R_Accu R_Amyg R_Caud R_Hipp R_Pall R_Puta R_Thal BrStem.
    structures = ['L_Amyg,L_Hipp,R_Amyg,R_Hipp']
    
    bashCommand = "run_first_all "
    param = '-d -b -s {} -i {} -o {}'.format(structures[0], Input, Output)
    
    startTime = datetime.now()

    os.system(bashCommand + param)
    
    print('Segmentation Done For {} / {} BrainMask.'.format(n+1, np.shape(no_proccessed_addresses)[0]))

    print('Elapsed Time: ', datetime.now() - startTime)

    preccessed_addresses.append(address)
    
    mdict = {'preccessed_addresses': preccessed_addresses}
    sio.savemat('preccessed_addresses.mat', mdict, appendmat=True, format='5',
                long_field_names=False, do_compression=False, oned_as='row')
