#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 16 11:31:05 2024

@author: dagi
"""

import os 
import nibabel as nib 
import cv2 
import shutil 
import numpy as np 

# In[]
src_path = "/media/Linux/Mallie_Dagmawi/PyTorch/data/Brain_MRI_Segmentation/archive/kaggle_3m/"
all_list = os.listdir(src_path)

dest_path = "/media/Linux/Mallie_Dagmawi/PyTorch/data/No Tumor/"

# In[] Clean all_list from non-directory elements
src_list = [item for item in all_list if os.path.isdir(os.path.join(src_path, item))]
with_mask = []
# In[] Iterate over src_list
for src_dir in src_list:
    # get path of the src_dir
    dir_path = src_path + src_dir + "/"
    dir_list = os.listdir(dir_path)
    # iterate over the src_dir
    for files in dir_list:
    # if the item doesn't have _mask in its name
        split_path = files.split("/")
        if "_mask" not in split_path[-1]:
            # set the full path to the file
            src_full_path = dir_path + files
            src_mask_path = src_full_path.replace(".tif", "_mask.tif")
            # read the corresponding mask file
            mask_img = cv2.imread(src_mask_path, cv2.IMREAD_GRAYSCALE)
            # Check if white pixels are present in mask_img if so it represents the presence of tumor
            if np.any(mask_img == 255): # tumor does exist in the mri image
                with_mask.append(files)    
                pass
            else:
                # set the path to destination
                shutil.copy(src_full_path, dest_path)
                # if the mask file doesn't contain any white pixel then no tumor is detected in that slice of MRI
# In[] Copy paste another sets of No tumor MRI images from another dataset
src_path_2 = "/media/Linux/Downloads/Brain Tumor/Brain Tumor Detection by MALICKS111/Training/no_tumor/"
src_list_2 = os.listdir(src_path_2)

# In[9 ] Iterate over the list and copy and paste to the dest folder
for items in src_list_2:
    # get the full path of the file
    full_path = src_path_2 + items
    # copy and paste the file 
    shutil.copy(full_path, dest_path)
    
# In[] T