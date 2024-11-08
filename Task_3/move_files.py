#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  7 14:27:02 2024

@author: dagi
"""

import os 
import shutil 
import random 

# In[]
src_glioma = "/media/Linux/Mallie_Dagmawi/PyTorch/data/Dataset/Model_3/Train/Glioma/"
src_meningioma = "/media/Linux/Mallie_Dagmawi/PyTorch/data/Dataset/Model_3/Train/Meningioma/"

dest_glioma = "/media/Linux/Mallie_Dagmawi/PyTorch/data/Reduced_dataset/Model_3/Glioma/"
dest_meningioma = "/media/Linux/Mallie_Dagmawi/PyTorch/data/Reduced_dataset/Model_3/Meningioma/"

# In[]
def return_list_path(path):
    list_path = os.listdir(path)
    full_path_list = []
    
    for each_file in list_path:
        full_path = path + each_file
        full_path_list.append(full_path)    
    return full_path_list 

# In[]
list_glioma = return_list_path(src_glioma)
list_meningioma = return_list_path(src_meningioma)

random.shuffle(list_glioma)
random.shuffle(list_meningioma)

# In[]
remove_glioma = int(0.15*len(list_glioma))
remove_meningioma = int(0.15*len(list_meningioma))

for index in range(len(list_meningioma)):
    if index < remove_meningioma:
        shutil.move(list_meningioma[index], dest_meningioma)
    else:
        break 
    
for index in range(len(list_glioma)):
    if index < remove_glioma:
        shutil.move(list_glioma[index], dest_glioma)
    else:
        break 